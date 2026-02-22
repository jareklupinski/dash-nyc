#!/usr/bin/env python3
"""
build.py — NYC Map static site generator.

Builds one or more apps from the apps/ directory.  Each app has its own
sources, templates, and static assets.  Shared code lives in lib/.

Usage:
    python build.py food                # build the food app
    python build.py food --cache        # use cached data if < 24h old
    python build.py food art            # build multiple apps
    python build.py --all               # build every app in apps/
    python build.py food --verbose      # debug logging
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

log = logging.getLogger("build")

ROOT = Path(__file__).parent
APPS_DIR = ROOT / "apps"
VENDOR = ROOT / "vendor"

CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


def _gzip_file(path: Path, level: int = 9) -> Path:
    """Write a .gz companion file for nginx gzip_static."""
    gz_path = path.parent / (path.name + ".gz")
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gz_path


# ---------------------------------------------------------------------------
# Map tile pre-download (Carto dark basemap for the NYC viewport)
# ---------------------------------------------------------------------------
_NYC_BBOX = (40.49, -74.27, 40.92, -73.68)  # SW_lat, SW_lng, NE_lat, NE_lng


def _download_map_tiles(dest_dir: Path, cache_dir: Path, zoom: int,
                        bbox: tuple = _NYC_BBOX) -> dict:
    """Download Carto dark basemap tiles for *bbox* at the given *zoom*.

    Returns ``{"<zoom>": {"x_min": …, "x_max": …, "y_min": …, "y_max": …}}``.
    Tiles are cached in *cache_dir* across builds.
    """
    import math
    import urllib.request

    def _tile(lat: float, lng: float, z: int) -> tuple[int, int]:
        n = 2 ** z
        x = int((lng + 180) / 360 * n)
        lat_r = math.radians(lat)
        y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
        return x, y

    sw_lat, sw_lng, ne_lat, ne_lng = bbox
    x_min, y_bot = _tile(sw_lat, sw_lng, zoom)
    x_max, y_top = _tile(ne_lat, ne_lng, zoom)

    # 1-tile buffer for edge panning
    x_min -= 1; x_max += 1; y_top -= 1; y_bot += 1

    subs = ["a", "b", "c"]
    dl = cached = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_top, y_bot + 1):
            tc = cache_dir / str(zoom) / str(x) / f"{y}.png"
            td = dest_dir / str(zoom) / str(x) / f"{y}.png"
            if not tc.exists():
                s = subs[(x + y) % 3]
                url = f"https://{s}.basemaps.cartocdn.com/dark_all/{zoom}/{x}/{y}.png"
                tc.parent.mkdir(parents=True, exist_ok=True)
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "dash-nyc-build/1.0 (+https://dash.nyc)"
                    })
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        tc.write_bytes(resp.read())
                    dl += 1
                except Exception as e:
                    log.warning("Tile %d/%d/%d failed: %s", zoom, x, y, e)
                    continue
            else:
                cached += 1
            td.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tc, td)

    total = (x_max - x_min + 1) * (y_bot - y_top + 1)
    log.info("Tiles z%d: %d fetched, %d cached of %d (x=%d\u2013%d y=%d\u2013%d)",
             zoom, dl, cached, total, x_min, x_max, y_top, y_bot)
    return {str(zoom): {"x_min": x_min, "x_max": x_max, "y_min": y_top, "y_max": y_bot}}


# ---------------------------------------------------------------------------
# Git SHA for traceability
# ---------------------------------------------------------------------------
def git_sha() -> str:
    """Return short git SHA of HEAD, or a content-hash fallback."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    # No git repo — hash tracked source files like git would
    import hashlib
    h = hashlib.sha1()
    for pattern in ("*.py", "*.html", "*.yaml", "*.css", "*.js", "*.json", "*.in", "*.md", "Makefile"):
        for p in sorted(ROOT.rglob(pattern)):
            rel = p.relative_to(ROOT)
            # Skip build outputs and caches
            if any(part in (".venv", "dist", "__pycache__", "node_modules", ".cache") for part in rel.parts):
                continue
            data = p.read_bytes()
            # Mimic git blob header: "blob <size>\0<content>"
            blob = f"blob {len(data)}\0".encode() + data
            h.update(hashlib.sha1(blob).digest())
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def load_cached(cache_dir: Path, source_name: str) -> list[dict] | None:
    cache_file = cache_dir / f"{source_name}.json"
    if not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age > CACHE_MAX_AGE_SECONDS:
        log.info("Cache for %s is stale (%.0fh old), refetching", source_name, age / 3600)
        return None
    log.info("Using cached data for %s (%.0fh old)", source_name, age / 3600)
    return json.loads(cache_file.read_text())


def save_cache(cache_dir: Path, source_name: str, venues: list[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{source_name}.json"
    cache_file.write_text(json.dumps(venues))
    log.info("Cached %d venues for %s", len(venues), source_name)


# ---------------------------------------------------------------------------
# Text normalization (shared across apps that do cross-source merging)
# ---------------------------------------------------------------------------
import re as _re
import math as _math

def _normalize(s: str) -> str:
    return _re.sub(r"\s+", " ", (s or "").strip().lower())


def _normalize_addr(s: str) -> str:
    s = _normalize(s)
    s = _re.sub(r"\baka\b.*", "", s)
    s = _re.sub(r"[.,#\-']", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    s = _re.sub(r"\s+(?:ste|suite|apt|unit|fl|floor|rm|room|#)\s*\S*$", "", s)
    s = _re.sub(r"\s+\d*[a-z]\d*$", "", s)
    s = _re.sub(r"\s+\d+$", "", s)
    s = _re.sub(r"\s+", " ", s).strip()
    s = _re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", s)
    _SUFFIXES = {
        "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane",
        "pl": "place", "rd": "road", "ct": "court", "cir": "circle",
        "ter": "terrace", "terr": "terrace", "pkwy": "parkway", "pky": "parkway",
        "hwy": "highway", "hgwy": "highway", "sq": "square", "tpke": "turnpike",
        "expy": "expressway", "expwy": "expressway",
        "e": "east", "w": "west", "n": "north", "s": "south", "saint": "st",
    }
    parts = s.split()
    parts = [_SUFFIXES.get(p, p) for p in parts]
    return " ".join(parts)


_BORO_ALIASES = {
    "manhattan": "manhattan", "new york": "manhattan", "ny": "manhattan",
    "brooklyn": "brooklyn", "bklyn": "brooklyn", "kings": "brooklyn",
    "queens": "queens",
    "bronx": "bronx", "the bronx": "bronx",
    "staten island": "staten island", "richmond": "staten island",
}

def _normalize_boro(s: str) -> str:
    return _BORO_ALIASES.get((s or "").strip().lower(), (s or "").strip().lower())


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    p = _math.pi / 180
    a = (0.5 - _math.cos((lat2 - lat1) * p) / 2
         + _math.cos(lat1 * p) * _math.cos(lat2 * p)
         * (1 - _math.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * _math.asin(_math.sqrt(a))


def make_venue_id(name: str, address: str = "", borough: str = "") -> str:
    """Create a URL-friendly, stable venue ID from name + address + borough."""
    slug = _re.sub(r'[^a-z0-9]+', '-', (name or '').lower().strip())
    slug = _re.sub(r'-+', '-', slug).strip('-')[:30]
    raw = f"{name}|{address}|{borough}".lower().strip()
    h = hashlib.sha256(raw.encode()).hexdigest()[:4]
    return f"{slug}-{h}" if slug else f"v-{h}"


# ---------------------------------------------------------------------------
# Cross-source merge (DOHMH + SLA → "both")
# ---------------------------------------------------------------------------
_RANGE_RE = _re.compile(r"^(\d+)\s+(\d+)\s+(.+)$")
_SINGLE_RE = _re.compile(r"^(\d+)\s+(.+)$")


def _make_combined(dohmh_v: dict, sla_v: dict) -> dict:
    combined = dict(dohmh_v)
    combined["source"] = "both"
    combined["sla_name"] = sla_v.get("name", "")
    all_tags = list(dict.fromkeys(dohmh_v.get("tags", []) + sla_v.get("tags", [])))
    combined["tags"] = all_tags
    combined_meta = dict(dohmh_v.get("meta", {}))
    combined_meta.update(sla_v.get("meta", {}))
    if combined_meta:
        combined["meta"] = combined_meta
    d_opened = dohmh_v.get("opened", "")
    s_opened = sla_v.get("opened", "")
    if d_opened and s_opened:
        combined["opened"] = min(d_opened, s_opened)
    elif s_opened:
        combined["opened"] = s_opened
    return combined


def _parse_range(raw_addr: str) -> tuple[int, int, str] | None:
    raw = raw_addr.strip()
    if _re.match(r"^\d+-\d+", raw):
        return None
    normed = _normalize_addr(raw)
    m = _RANGE_RE.match(normed)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        street = m.group(3)
        if hi >= lo and (hi - lo) <= 30:
            return (lo, hi, street)
    return None


def _parse_single_number(normed_addr: str) -> tuple[int, str] | None:
    m = _SINGLE_RE.match(normed_addr)
    if m:
        try:
            return (int(m.group(1)), m.group(2))
        except ValueError:
            pass
    return None


def merge_cross_source(venues: list[dict]) -> tuple[list[dict], dict]:
    """Merge venues found in both DOHMH and SLA into source='both'.

    Pass 1 — exact match on normalized address + borough.
    Pass 2 — range match (SLA ranges containing DOHMH number).
    Pass 3 — geo-proximity fallback (≤30m + name similarity ≥0.35).
    """
    from collections import defaultdict

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in venues:
        addr = _normalize_addr(v.get("address", ""))
        boro = _normalize_boro(v.get("borough", ""))
        if addr:
            by_key[(addr, boro)].append(v)
        else:
            by_key[("_noaddr_" + str(id(v)), "")].append(v)

    merged: list[dict] = []
    unmatched_sla: list[dict] = []
    unmatched_dohmh: list[dict] = []
    pass1_count = 0

    for key, group in by_key.items():
        sources_in_group = {v["source"] for v in group}
        if "dohmh" in sources_in_group and "sla" in sources_in_group:
            dohmh_v = next(v for v in group if v["source"] == "dohmh")
            sla_v = next(v for v in group if v["source"] == "sla")
            merged.append(_make_combined(dohmh_v, sla_v))
            pass1_count += 1
            for v in group:
                if v is not dohmh_v and v is not sla_v:
                    merged.append(v)
        else:
            for v in group:
                if v["source"] == "sla":
                    unmatched_sla.append(v)
                elif v["source"] == "dohmh":
                    unmatched_dohmh.append(v)
                else:
                    merged.append(v)

    log.info("  Pass 1 (exact address): %d merges", pass1_count)

    # Pass 2: range matching
    dohmh_by_street: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for v in unmatched_dohmh:
        normed = _normalize_addr(v.get("address", ""))
        boro = _normalize_boro(v.get("borough", ""))
        parsed = _parse_single_number(normed)
        if parsed:
            num, street = parsed
            dohmh_by_street[(street, boro)].append((num, v))

    range_merged_sla: set[int] = set()
    range_merged_dohmh: set[int] = set()

    for sla_v in unmatched_sla:
        rng = _parse_range(sla_v.get("address", ""))
        if not rng:
            continue
        lo, hi, street = rng
        boro = _normalize_boro(sla_v.get("borough", ""))
        candidates = dohmh_by_street.get((street, boro), [])
        for num, dohmh_v in candidates:
            if lo <= num <= hi and id(dohmh_v) not in range_merged_dohmh:
                merged.append(_make_combined(dohmh_v, sla_v))
                range_merged_sla.add(id(sla_v))
                range_merged_dohmh.add(id(dohmh_v))
                break

    log.info("  Pass 2 (address range): %d merges", len(range_merged_sla))

    still_sla = [v for v in unmatched_sla if id(v) not in range_merged_sla]
    still_dohmh = [v for v in unmatched_dohmh if id(v) not in range_merged_dohmh]

    # Pass 3: geo-proximity
    GEO_RADIUS_M = 30
    GEO_NAME_THRESHOLD = 0.35
    CELL = 0.0003

    dohmh_geo_grid: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for v in still_dohmh:
        lat, lng = v.get("lat"), v.get("lng")
        if lat and lng:
            cell = (int(lat / CELL), int(lng / CELL))
            dohmh_geo_grid[cell].append(v)

    geo_merged_sla: set[int] = set()
    geo_merged_dohmh: set[int] = set()

    for sla_v in still_sla:
        lat, lng = sla_v.get("lat"), sla_v.get("lng")
        if not (lat and lng):
            continue
        cell = (int(lat / CELL), int(lng / CELL))
        best_dist = float("inf")
        best_match = None
        sla_name = _normalize(sla_v.get("name", ""))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dv in dohmh_geo_grid.get((cell[0] + di, cell[1] + dj), []):
                    if id(dv) in geo_merged_dohmh:
                        continue
                    d = _haversine_m(lat, lng, dv["lat"], dv["lng"])
                    if d < best_dist:
                        dohmh_name = _normalize(dv.get("name", ""))
                        sim = SequenceMatcher(None, sla_name, dohmh_name).ratio()
                        if sim >= GEO_NAME_THRESHOLD:
                            best_dist = d
                            best_match = dv
        if best_dist <= GEO_RADIUS_M and best_match is not None:
            merged.append(_make_combined(best_match, sla_v))
            geo_merged_sla.add(id(sla_v))
            geo_merged_dohmh.add(id(best_match))

    log.info("  Pass 3 (geo ≤%dm): %d merges", GEO_RADIUS_M, len(geo_merged_sla))

    for v in still_sla:
        if id(v) not in geo_merged_sla:
            merged.append(v)
    for v in still_dohmh:
        if id(v) not in geo_merged_dohmh:
            merged.append(v)

    total_merges = pass1_count + len(range_merged_sla) + len(geo_merged_sla)
    stats = {
        "pre_merge": len(venues),
        "pass1": pass1_count,
        "pass2": len(range_merged_sla),
        "pass3": len(geo_merged_sla),
        "total_merges": total_merges,
        "post_merge": len(merged),
    }
    return merged, stats


# ---------------------------------------------------------------------------
# Borough bbox validation
# ---------------------------------------------------------------------------
_BORO_BOUNDS = {
    "manhattan":      (40.698, 40.882, -74.025, -73.907),
    "brooklyn":       (40.566, 40.740, -74.045, -73.830),
    "queens":         (40.540, 40.812, -73.963, -73.700),
    "bronx":          (40.785, 40.917, -73.935, -73.748),
    "staten island":  (40.490, 40.652, -74.260, -74.050),
}


def validate_coords(venues: list[dict]) -> int:
    """Drop coords that fall outside the venue's borough bbox. Returns count dropped."""
    bad = 0
    for v in venues:
        lat, lng = v.get("lat"), v.get("lng")
        if not (lat and lng):
            continue
        boro = _normalize_boro(v.get("borough", ""))
        bounds = _BORO_BOUNDS.get(boro)
        if not bounds:
            continue
        lat_lo, lat_hi, lng_lo, lng_hi = bounds
        if not (lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi):
            del v["lat"]
            del v["lng"]
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def apply_overrides(venues: list[dict], overrides_file: Path) -> int:
    """Apply manual overrides. Returns count applied."""
    if not overrides_file.exists():
        return 0
    data = json.loads(overrides_file.read_text())
    entries = data.get("overrides", [])
    drop_keys: set[tuple[str, str, str]] = set()
    for o in entries:
        if o.get("action") == "drop_coords":
            drop_keys.add((
                o["name"].strip().lower(),
                o["address"].strip().lower(),
                _normalize_boro(o.get("borough", "")),
            ))
    applied = 0
    for v in venues:
        key = (
            v.get("name", "").strip().lower(),
            v.get("address", "").strip().lower(),
            _normalize_boro(v.get("borough", "")),
        )
        if key in drop_keys and "lat" in v and "lng" in v:
            del v["lat"]
            del v["lng"]
            applied += 1
    return applied


# ---------------------------------------------------------------------------
# Cross-ref stamping
# ---------------------------------------------------------------------------
def stamp_crossref(venues: list[dict], app_dir: Path) -> tuple[int, int, dict]:
    """Stamp crossref flags from the shared crossref DB. Returns (checked, coords_upgraded, stats)."""
    from lib.crossref import venue_key as xref_key, get_flags, get_stats as xref_stats, init_db as xref_init
    import lib.crossref as crossref

    if not crossref.DB_PATH.exists():
        log.info("Cross-ref: no DB at %s — skipping", crossref.DB_PATH)
        return 0, 0, {}

    xconn = xref_init()
    flags = get_flags(xconn)
    xs = xref_stats(xconn)
    xconn.close()

    checked = 0
    coords_upgraded = 0
    for v in venues:
        k = xref_key(v["name"], v.get("address", ""), v.get("borough", ""))
        f = flags.get(k)
        if not f:
            continue
        v["xr_y"] = f["yelp"]
        v["xr_g"] = f["google"]
        if f.get("yelp_reviews") is not None:
            v["yr"] = f["yelp_reviews"]
        if f.get("yelp_rating") is not None:
            v["yrt"] = f["yelp_rating"]
        if f.get("google_reviews") is not None:
            v["gr"] = f["google_reviews"]
        if f.get("google_rating") is not None:
            v["grt"] = f["google_rating"]
        if f.get("yelp_categories"):
            v["yelp_cats"] = f["yelp_categories"]
        v["xr_ot"] = f.get("opentable", "unchecked")
        if f.get("opentable_reviews") is not None:
            v["otr"] = f["opentable_reviews"]
        if f.get("opentable_rating") is not None:
            v["otrt"] = f["opentable_rating"]
        if f.get("opentable_url"):
            v["ot_url"] = f["opentable_url"]

        boro = _normalize_boro(v.get("borough", ""))
        bounds = _BORO_BOUNDS.get(boro)
        for src, slat, slng in (("google", f.get("google_lat"), f.get("google_lng")),
                                ("yelp",   f.get("yelp_lat"),   f.get("yelp_lng"))):
            if not slat or not slng:
                continue
            if not bounds or (bounds[0] <= slat <= bounds[1] and bounds[2] <= slng <= bounds[3]):
                v["lat"] = slat
                v["lng"] = slng
                coords_upgraded += 1
                break
        checked += 1

    return checked, coords_upgraded, xs


# ---------------------------------------------------------------------------
# Dietary tagging
# ---------------------------------------------------------------------------
def apply_diet_tags(venues: list[dict], app_dir: Path, use_cache: bool) -> tuple[dict, dict, int, int]:
    """Apply dietary tags from authoritative + supplementary sources.
    Returns (diet_counts, diet_source_stats, hms_matched, knm_matched).
    """
    try:
        from lib.diet_sources import fetch_hms_halal, fetch_knm_kosher
    except ImportError:
        log.info("Diet sources not available for this app — skipping")
        return {}, {}, 0, 0

    hms_entries = fetch_hms_halal(use_cache=use_cache)
    knm_entries = fetch_knm_kosher(use_cache=use_cache)

    hms_by_addr: dict[tuple[str, str], dict] = {}
    hms_by_name: dict[tuple[str, str], dict] = {}
    for h in hms_entries:
        na = _normalize_addr(h.get("address", "").split(",")[0])
        boro = h.get("borough", "")
        nn = _normalize(h.get("name", ""))
        if na and boro:
            hms_by_addr[(na, boro)] = h
        if nn and boro:
            hms_by_name[(nn, boro)] = h

    _KNM_CELL = 0.0003
    knm_geo: dict[tuple[int, int], list[dict]] = {}
    knm_by_name: dict[str, list[dict]] = {}
    for k in knm_entries:
        lat, lng = k.get("lat"), k.get("lng")
        if lat and lng:
            cell = (int(lat / _KNM_CELL), int(lng / _KNM_CELL))
            knm_geo.setdefault(cell, []).append(k)
        nn = _normalize(k.get("name", ""))
        if nn:
            knm_by_name.setdefault(nn, []).append(k)

    def _match_hms(v):
        boro = _normalize_boro(v.get("borough", ""))
        if not boro:
            return None
        na = _normalize_addr(v.get("address", ""))
        hit = hms_by_addr.get((na, boro))
        if hit:
            return hit
        nn = _normalize(v.get("name", ""))
        return hms_by_name.get((nn, boro))

    def _match_knm(v):
        lat, lng = v.get("lat"), v.get("lng")
        if lat and lng:
            cell = (int(lat / _KNM_CELL), int(lng / _KNM_CELL))
            v_name = _normalize(v.get("name", ""))
            best_dist = float("inf")
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for k in knm_geo.get((cell[0] + di, cell[1] + dj), []):
                        d = _haversine_m(lat, lng, k["lat"], k["lng"])
                        if d < best_dist and d <= 80:
                            kn = _normalize(k.get("name", ""))
                            sim = SequenceMatcher(None, v_name, kn).ratio()
                            if sim >= 0.45:
                                best_dist = d
                                best = k
            if best:
                return best
        nn = _normalize(v.get("name", ""))
        hits = knm_by_name.get(nn, [])
        return hits[0] if len(hits) == 1 else None

    _YELP_DIET = {
        "vegan": "vegan", "vegetarian": "vegetarian", "veganraw": "vegan",
        "raw_food": "vegan", "gluten_free": "gluten-free",
    }
    _DOHMH_DIET = {"Vegetarian": "vegetarian", "Vegan": "vegan"}

    diet_counts: dict[str, int] = {}
    diet_source_counts: dict[str, dict[str, int]] = {}
    hms_matched = 0
    knm_matched = 0

    for v in venues:
        diets: dict[str, str] = {}

        hms_hit = _match_hms(v)
        if hms_hit:
            diets["halal"] = "HMS USA"
            hms_matched += 1

        knm_hit = _match_knm(v)
        if knm_hit:
            diets["kosher"] = "KosherNearMe"
            knm_matched += 1

        cuisine = v.get("cuisine", "")
        dt = _DOHMH_DIET.get(cuisine)
        if dt and dt not in diets:
            diets[dt] = "DOHMH"
        cl = cuisine.lower()
        if "halal" in cl and "halal" not in diets:
            diets["halal"] = "DOHMH"
        if "kosher" in cl and "kosher" not in diets:
            diets["kosher"] = "DOHMH"
        if cuisine == "Jewish/Kosher" and "kosher" not in diets:
            diets["kosher"] = "DOHMH"

        for yc in v.get("yelp_cats", []):
            dt = _YELP_DIET.get(yc)
            if dt and dt not in diets:
                diets[dt] = "Yelp"
            if yc == "halal" and "halal" not in diets:
                diets["halal"] = "Yelp"
            if yc == "kosher" and "kosher" not in diets:
                diets["kosher"] = "Yelp"

        if "vegan" in diets and "vegetarian" not in diets:
            diets["vegetarian"] = diets["vegan"]

        if diets:
            v["diet"] = sorted(diets.keys())
            v["diet_src"] = diets
            for d, src in diets.items():
                diet_counts[d] = diet_counts.get(d, 0) + 1
                diet_source_counts.setdefault(d, {})
                diet_source_counts[d][src] = diet_source_counts[d].get(src, 0) + 1

    return diet_counts, diet_source_counts, hms_matched, knm_matched


# ---------------------------------------------------------------------------
# App loader
# ---------------------------------------------------------------------------
def load_app_config(app_id: str) -> dict:
    """Load and return app.yaml for the given app ID."""
    app_dir = APPS_DIR / app_id
    config_file = app_dir / "app.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"App config not found: {config_file}")
    return yaml.safe_load(config_file.read_text())


def list_apps() -> list[str]:
    """Return list of app IDs found in apps/."""
    return sorted(
        d.name for d in APPS_DIR.iterdir()
        if d.is_dir() and (d / "app.yaml").exists()
    )


# ---------------------------------------------------------------------------
# Pre-render venue info panel HTML (per-app)
# ---------------------------------------------------------------------------
from html import escape as _html_esc
from datetime import timedelta
from urllib.parse import quote as _url_quote

_ONE_MONTH_AGO = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

_DPR_TAGS = {
    "food_cart": "Food Cart",
    "mobile_food_truck": "Food Truck",
    "snack_bar": "Snack Bar",
    "specialty_cart": "Specialty Cart",
    "restaurant": "Restaurant",
    "breakfast_cart": "Breakfast Cart",
}


def _vi_row(label: str, val: str, cls: str = "") -> str:
    if not val:
        return ""
    return f'<div class="vi-row"><span class="vi-label">{label}</span><span class="{cls}">{val}</span></div>'


def _vi_actions(back_only: bool = False, link_url: str = "", link_label: str = "Maps ↗", share_id: str = "") -> str:
    """Render Back + Share + Link buttons below venue info."""
    h = '<div class="vi-actions">'
    h += '<button class="vi-back">Back</button>'
    if share_id:
        h += f'<button class="vi-share-btn" data-vid="{_html_esc(share_id)}">\U0001f4cb Link</button>'
    if link_url and not back_only:
        h += f'<a class="vi-link-btn" href="{_html_esc(link_url)}" target="_blank" rel="noopener">{link_label}</a>'
    h += '</div>'
    return h


def _eat_badges(v: dict) -> list[dict]:
    badges = []
    if v.get("xr_g") == "found" and (v.get("gr") or 0) < 100 and (v.get("grt") or 0) > 4:
        badges.append({"type": "google-gem"})
    if v.get("xr_y") == "found" and (v.get("yr") or 0) < 100 and (v.get("yrt") or 0) > 4:
        badges.append({"type": "yelp-gem"})
    if v.get("opened") and v["opened"] >= _ONE_MONTH_AGO:
        badges.append({"type": "new-venue"})
    if v.get("xr_y") == "not_found":
        badges.append({"type": "not-on-yelp"})
    if v.get("xr_g") == "not_found":
        badges.append({"type": "not-on-google"})
    if v.get("source") == "dpr" and v.get("tags"):
        for t in v["tags"]:
            if t in _DPR_TAGS:
                badges.append({"type": "dpr", "tag": t})
    return badges


# ---------------------------------------------------------------------------
# Unified filter computation — single source of truth for all templates
# ---------------------------------------------------------------------------

_FILTER_LABELS = {
    "halal": "Halal", "kosher": "Kosher", "vegan": "Vegan",
    "vegetarian": "Vegetarian", "gluten_free": "Gluten-Free",
    "google_gem": "Google Gems", "yelp_gem": "Yelp Gems",
    "new_venue": "New", "not_on_yelp": "Not on Yelp",
    "not_on_google": "Not on Google", "dpr_food": "Park Food",
    "bar": "Bars", "coffee": "Coffee & Tea", "juice": "Juice & Frozen",
    "farmers_market": "Farmers Market", "grocery": "Grocery",
    "grocery_ebt": "Grocery (EBT)", "positive": "Good News",
    "negative": "Bad News",
}

# Ordered extra filter keys per app (only shown if venues exist)
_APP_EXTRA_FILTERS: dict[str, list[str]] = {
    "eat":   ["google_gem", "yelp_gem", "new_venue", "not_on_yelp", "not_on_google", "dpr_food"],
    "drink": ["bar", "coffee", "juice", "new_venue"],
    "food":  ["farmers_market", "grocery", "grocery_ebt"],
    "news":  ["positive", "negative"],
}


def _compute_venue_filters(
    all_venues: list[dict], app_id: str, app_conf: dict, all_diets: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Pre-compute ``_ft`` (filter tags) on every venue and return (filter_list, label_map).

    This is the **single source of truth** for filter membership.  Both the
    SPA portal (dash template) and standalone per-app templates consume the
    same pre-computed ``_ft`` arrays, so filter logic can never diverge.
    """
    has_xref = app_conf.get("crossref", {})
    has_diet = app_conf.get("diet", {}).get("enabled", False)

    for v in all_venues:
        ft: list[str] = []

        # ── Diet tags (eat) ────────────────────────────────────
        if has_diet and v.get("diet"):
            ft.extend(v["diet"])

        # ── Cross-ref hidden gems (low reviews + high rating) ──
        if has_xref.get("google"):
            if (v.get("xr_g") == "found"
                    and (v.get("gr") or 0) < 100
                    and (v.get("grt") or 0) > 4):
                ft.append("google_gem")
            if v.get("xr_g") == "not_found":
                ft.append("not_on_google")
        if has_xref.get("yelp"):
            if (v.get("xr_y") == "found"
                    and (v.get("yr") or 0) < 100
                    and (v.get("yrt") or 0) > 4):
                ft.append("yelp_gem")
            if v.get("xr_y") == "not_found":
                ft.append("not_on_yelp")

        # ── New venue (opened within 30 days) ─────────────────
        if v.get("opened") and v["opened"] >= _ONE_MONTH_AGO:
            ft.append("new_venue")

        # ── DPR park food vendors ──────────────────────────────
        if v.get("source") == "dpr" and v.get("tags"):
            if any(t in _DPR_TAGS for t in v["tags"]):
                ft.append("dpr_food")

        # ── Drink type filters ─────────────────────────────────
        if app_id == "drink":
            if v.get("source") == "sla":
                ft.append("bar")
            tags = v.get("tags") or []
            if any(t in tags for t in ("coffee_tea", "bubble_tea")):
                ft.append("coffee")
            if any(t in tags for t in ("juice", "frozen")):
                ft.append("juice")

        # ── Food source filters ────────────────────────────────
        if app_id == "food":
            if v.get("source") == "greenmarket":
                ft.append("farmers_market")
            elif v.get("source") == "grocery":
                ft.append("grocery")
                if v.get("tags") and "ebt" in v["tags"]:
                    ft.append("grocery_ebt")

        # ── News sentiment ─────────────────────────────────────
        if app_id == "news":
            sent = (v.get("meta") or {}).get("sentiment")
            if sent == "POSITIVE":
                ft.append("positive")
            elif sent == "NEGATIVE":
                ft.append("negative")

        # ── Raw tags for non-diet apps (topic tags, source tags, etc.) ──
        # Exclude drink/food where type filters replace raw tags.
        if not has_diet and app_id not in ("drink", "food"):
            for rt in (v.get("tags") or []):
                if rt not in ft:
                    ft.append(rt)

        v["_ft"] = ft

    # ── Build ordered filter list (only include tags with venues) ──
    ft_counts: dict[str, int] = {}
    for v in all_venues:
        for t in v.get("_ft", []):
            ft_counts[t] = ft_counts.get(t, 0) + 1

    filters: list[str] = []
    # Diet filters first
    if has_diet:
        for d in all_diets:
            if ft_counts.get(d, 0) > 0:
                filters.append(d)

    # App-specific extras (in presentation order)
    extras = _APP_EXTRA_FILTERS.get(app_id, [])
    for f in extras:
        if f not in filters and ft_counts.get(f, 0) > 0:
            filters.append(f)

    # Remaining raw tags (sorted): include for apps without explicit extras,
    # or for news which mixes explicit extras (positive/negative) with
    # dynamic topic tags from the raw data.
    if app_id not in _APP_EXTRA_FILTERS or app_id == "news":
        for t in sorted(ft_counts.keys()):
            if t not in filters and ft_counts.get(t, 0) > 0:
                filters.append(t)

    # Label map — only for filters actually shown
    filter_labels: dict[str, str] = {}
    for f in filters:
        if f in _FILTER_LABELS:
            filter_labels[f] = _FILTER_LABELS[f]
        else:
            filter_labels[f] = f.replace("_", " ").title()

    log.info("Filters for %s: %s", app_id, filters)
    return filters, filter_labels


_DIET_SOURCES = {
    "HMS": {
        "icon": '<svg viewBox="0 0 16 16" width="12" height="12"><circle cx="8" cy="8" r="7" fill="#065f46" stroke="#065f46" stroke-width="1"/><text x="8" y="11.5" text-anchor="middle" fill="#fff" font-size="10" font-weight="700" font-family="sans-serif">H</text></svg>',
        "url": "https://www.hmsusa.org/certified-listing",
        "label": "HMS USA certified halal",
    },
    "KosherNearMe": {
        "icon": '<svg viewBox="0 0 16 16" width="12" height="12"><circle cx="8" cy="8" r="7" fill="#1e40af" stroke="#1e40af" stroke-width="1"/><text x="8" y="11.5" text-anchor="middle" fill="#fff" font-size="10" font-weight="700" font-family="sans-serif">K</text></svg>',
        "url": "https://koshernear.me/",
        "label": "KosherNearMe listing",
    },
    "DOHMH": {
        "icon": '<svg viewBox="0 0 16 16" width="12" height="12"><rect x="1" y="1" width="14" height="14" rx="3" fill="#6b7280" stroke="#6b7280" stroke-width="1"/><text x="8" y="11.5" text-anchor="middle" fill="#fff" font-size="9" font-weight="700" font-family="sans-serif">D</text></svg>',
        "url": "https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j",
        "label": "DOHMH cuisine classification",
    },
}


def _diet_tag_html(diet: str, src_name: str) -> str:
    e = _html_esc
    meta = _DIET_SOURCES.get(src_name or "")
    if not meta:
        return f'<span class="diet-tag">{e(diet)}</span>'
    if meta.get("url"):
        inner = f'<a href="{meta["url"]}" target="_blank" rel="noopener" class="diet-src-link" title="{e(meta["label"])}">{meta["icon"]}</a>'
    else:
        inner = f'<span class="diet-src-icon" title="{e(meta["label"])}">{meta["icon"]}</span>'
    return f'<span class="diet-tag">{e(diet)}{inner}</span>'


def _render_eat_info(v: dict) -> str:
    e = _html_esc
    badges = _eat_badges(v)
    h = '<div class="vi-term">'
    h += _vi_row("NAME", e(v.get("name", "")), "vi-name")
    sla = v.get("sla_name", "")
    if sla and sla.lower() != v.get("name", "").lower():
        h += _vi_row("AKA", e(sla), "vi-addr")
    if v.get("cuisine"):
        h += _vi_row("TYPE", e(v["cuisine"]), "vi-meta")
    diets = v.get("diet") or []
    if diets:
        diet_src = v.get("diet_src") or {}
        tags = [_diet_tag_html(d, diet_src.get(d, "")) for d in diets]
        h += _vi_row("DIET", " ".join(tags), "vi-tags")
    if v.get("address"):
        h += _vi_row("ADDR", e(v["address"]), "vi-addr")
    if v.get("borough"):
        h += _vi_row("BORO", e(v["borough"]), "vi-addr")
    if v.get("grade"):
        h += _vi_row("GRDE", e(v["grade"]), "vi-meta")
    if v.get("phone"):
        h += _vi_row("TEL", e(v["phone"]), "vi-addr")
    h += _vi_row("SRC", e(v.get("source", "")), "vi-src")
    for badge in badges:
        bt = badge["type"]
        if bt == "google-gem":
            h += _vi_row("FLAG", f'<span style="color:#2563eb">G</span> Google gem ({v.get("grt", "")}★, {v.get("gr", "")} reviews)', "")
        elif bt == "yelp-gem":
            h += _vi_row("FLAG", f'<span style="color:#dc2626">Y</span> Yelp gem ({v.get("yrt", "")}★, {v.get("yr", "")} reviews)', "")
        elif bt == "new-venue":
            h += _vi_row("FLAG", f'<span style="color:#f59e0b">⏱</span> Opened {e(v.get("opened", ""))}', "")
        elif bt == "not-on-yelp":
            h += _vi_row("FLAG", '<span style="color:#dc2626">★</span> Not on Yelp', "")
        elif bt == "not-on-google":
            h += _vi_row("FLAG", '<span style="color:#2563eb">★</span> Not on Google', "")
        elif bt == "dpr":
            h += _vi_row("FLAG", f'<span style="color:#34d399">🌳</span> Park {e(_DPR_TAGS.get(badge["tag"], ""))}', "")
    rv = []
    if v.get("yr") is not None:
        rv.append(f'Yelp: {v.get("yrt", "?")}★ ({v["yr"]})')
    if v.get("gr") is not None:
        rv.append(f'Google: {v.get("grt", "?")}★ ({v["gr"]})')
    if rv:
        h += _vi_row("REVW", " · ".join(rv), "vi-reviews")
    h += "</div>"
    h += _vi_actions(share_id=v.get("id", ""))
    return h


def _render_drink_info(v: dict) -> str:
    e = _html_esc
    h = '<div class="vi-term">'
    h += _vi_row("NAME", e(v.get("name", "")), "vi-name")
    if v.get("cuisine"):
        h += _vi_row("TYPE", e(v["cuisine"]), "vi-meta")
    if v.get("address"):
        h += _vi_row("ADDR", e(v["address"]), "vi-addr")
    if v.get("borough"):
        h += _vi_row("BORO", e(v["borough"]), "vi-addr")
    if v.get("grade"):
        h += _vi_row("GRDE", e(v["grade"]), "vi-meta")
    if v.get("phone"):
        h += _vi_row("TEL", e(v["phone"]), "vi-addr")
    src = v.get("source", "")
    src_label = "Bar (SLA)" if src == "sla" else "Coffee/Tea (DOHMH)" if src == "coffee" else src
    h += _vi_row("SRC", e(src_label), "vi-src")
    if v.get("opened") and v["opened"] >= _ONE_MONTH_AGO:
        h += _vi_row("FLAG", f'<span style="color:#f59e0b">⏱</span> Opened {e(v["opened"])}', "")
    meta = v.get("meta") or {}
    if meta.get("license_type"):
        h += _vi_row("LIC", e(meta["license_type"]), "vi-meta")
    h += "</div>"
    h += _vi_actions(share_id=v.get("id", ""))
    return h


def _render_food_info(v: dict) -> str:
    e = _html_esc
    h = '<div class="vi-term">'
    h += _vi_row("NAME", e(v.get("name", "")), "vi-name")
    if v.get("address"):
        h += _vi_row("ADDR", e(v["address"]), "vi-addr")
    if v.get("borough"):
        h += _vi_row("BORO", e(v["borough"]), "vi-addr")
    if v.get("phone"):
        h += _vi_row("TEL", e(v["phone"]), "vi-addr")
    meta = v.get("meta") or {}
    tags = v.get("tags") or []
    if v.get("source") == "greenmarket" or "farmers-market" in tags:
        sched = meta.get("schedule", "")
        ebt = meta.get("ebt")
        info_parts = ["Farmers Market"]
        if sched:
            info_parts.append(sched)
        if ebt:
            info_parts.append("EBT")
        h += _vi_row("INFO", " · ".join(info_parts), "vi-tags")
    elif v.get("source") == "grocery":
        et = meta.get("estab_type", "")
        if "B" in et:
            label = "Bakery"
        elif "C" in et:
            label = "Deli/Grocery"
        else:
            label = "Food Store"
        if meta.get("ebt"):
            label += " · EBT"
        h += _vi_row("INFO", label, "vi-tags")
    h += _vi_row("SRC", e(v.get("source", "")), "vi-src")
    h += "</div>"
    h += _vi_actions(share_id=v.get("id", ""))
    return h


def _render_news_info(v: dict) -> str:
    e = _html_esc
    meta = v.get("meta") or {}
    h = '<div class="vi-term">'
    lnk = meta.get("link", "")
    if lnk:
        h += _vi_row("TITLE", f'<a href="{e(lnk)}" target="_blank" rel="noopener">{e(v.get("name", ""))}</a>', "vi-name vi-link")
    else:
        h += _vi_row("TITLE", e(v.get("name", "")), "vi-name")
    if meta.get("summary"):
        h += _vi_row("  ...", e(meta["summary"]), "vi-addr")
    sentiment = meta.get("sentiment", "")
    if sentiment:
        pos = sentiment == "POSITIVE"
        score = meta.get("score")
        pct = f" {round(score * 100)}%" if score is not None else ""
        icon = "☀ " if pos else "☁ "
        h += _vi_row("MOOD", f'{icon}{sentiment.lower()}{pct}', f'vi-sentiment {"pos" if pos else "neg"}')
    if v.get("cuisine"):
        h += _vi_row("TOPIC", e(v["cuisine"]), "vi-tags")
    fp = []
    if meta.get("feed"):
        fp.append(e(meta["feed"]))
    if meta.get("published"):
        fp.append(e(meta["published"]))
    if fp:
        h += _vi_row("FROM", " · ".join(fp), "vi-src")
    if meta.get("matched_location"):
        h += _vi_row("LOC", f'📍 {e(meta["matched_location"])}', "vi-addr")
    if lnk:
        h += _vi_row("LINK", f'<a href="{e(lnk)}" target="_blank" rel="noopener">Read article →</a>', "vi-link")
    h += "</div>"
    h += _vi_actions(link_url=lnk or "", link_label="Article ↗", share_id=v.get("id", ""))
    return h


def _render_generic_info(v: dict) -> str:
    """Generic venue info renderer for apps without a custom renderer."""
    e = _html_esc
    h = '<div class="vi-term">'
    h += _vi_row("NAME", e(v.get("name", "")), "vi-name")
    if v.get("address"):
        h += _vi_row("ADDR", e(v["address"]), "vi-addr")
    if v.get("borough"):
        h += _vi_row("BORO", e(v["borough"]), "vi-addr")
    if v.get("phone"):
        h += _vi_row("TEL", e(v["phone"]), "vi-addr")
    if v.get("cuisine"):
        h += _vi_row("TYPE", e(v["cuisine"]), "vi-tags")
    tags = v.get("tags") or []
    if tags:
        h += _vi_row("TAGS", e(", ".join(tags)), "vi-tags")
    h += _vi_row("SRC", e(v.get("source", "")), "vi-src")
    h += "</div>"
    h += _vi_actions(share_id=v.get("id", ""))
    return h


_VENUE_RENDERERS = {
    "eat": _render_eat_info,
    "drink": _render_drink_info,
    "food": _render_food_info,
    "dash": _render_news_info,
    "news": _render_news_info,
    "art": _render_generic_info,
    "shop": _render_generic_info,
    "theater": _render_generic_info,
    "play": _render_generic_info,
    "movies": _render_generic_info,
    "music": _render_generic_info,
    "make": _render_generic_info,
    "relax": _render_generic_info,
    "pee": _render_generic_info,
    "weather": _render_generic_info,
    "today": _render_generic_info,
    "vibe": _render_generic_info,
}


def render_all_venue_infos(venues: list[dict], app_id: str) -> list[str]:
    """Pre-render venue info panel HTML for all venues."""
    renderer = _VENUE_RENDERERS.get(app_id)
    if not renderer:
        return []
    return [renderer(v) for v in venues]


# ---------------------------------------------------------------------------
# Build a single app
# ---------------------------------------------------------------------------
def build_app(app_id: str, use_cache: bool = False, selected_sources: set[str] | None = None) -> Path:
    """Build a single app. Returns the dist directory path."""
    config = load_app_config(app_id)
    app_conf = config["app"]
    app_dir = APPS_DIR / app_id
    dist_dir = ROOT / "dist" / app_id
    cache_dir = app_dir / ".cache"

    sha = git_sha()
    log.info("=" * 60)
    log.info("Building app: %s (%s) [%s]", app_conf["title"], app_id, sha)
    log.info("=" * 60)

    # Discover sources from app config's sources list (falls back to app's sources/ dir)
    from lib.source_base import discover_sources, discover_sources_from_files, DataSource

    source_names = app_conf.get("sources", [])
    if source_names:
        # Resolve each source name to a file path:
        # 1. Check app-local sources/ dir first
        # 2. Fall back to top-level sources/ dir
        source_files: list[Path] = []
        for sname in source_names:
            local = app_dir / "sources" / f"{sname}.py"
            shared = ROOT / "sources" / f"{sname}.py"
            if local.exists():
                source_files.append(local)
            elif shared.exists():
                source_files.append(shared)
            else:
                log.error("Source '%s' not found for app %s (checked %s and %s)",
                          sname, app_id, local, shared)
        sources = discover_sources_from_files(source_files)
    else:
        # Legacy: discover all sources from app's sources/ directory
        sources_dir = app_dir / "sources"
        sources = discover_sources(sources_dir)

    # Set cache_dir on each source so they store internal caches in the app's .cache/
    for source in sources:
        source.cache_dir = cache_dir

    if selected_sources:
        sources = [s for s in sources if s.name in selected_sources]

    if not sources and source_names:
        log.error("No data sources found for app %s!", app_id)
        return dist_dir
    elif not sources:
        log.info("App %s has no data sources — building static-only", app_id)

    log.info("Active sources: %s", ", ".join(s.name for s in sources))

    # Fetch from all sources
    all_venues: list[dict] = []
    source_meta: list[dict] = []
    pipeline_stats: list[dict] = []

    for source in sources:
        log.info("=== Fetching: %s — %s ===", source.name, source.description)

        cached = load_cached(cache_dir, source.name) if use_cache else None
        from_cache = cached is not None
        if cached is not None:
            venue_dicts = cached
        else:
            try:
                venues = source.fetch()
                venue_dicts = [v.to_dict() for v in venues]
                save_cache(cache_dir, source.name, venue_dicts)
            except Exception:
                log.exception("Failed to fetch from %s", source.name)
                cached = load_cached(cache_dir, source.name) if not use_cache else None
                if cached:
                    log.warning("Using stale cache for %s as fallback", source.name)
                    venue_dicts = cached
                    from_cache = True
                else:
                    continue

        # Determine cache / refresh timestamps
        cache_file = cache_dir / f"{source.name}.json"
        cache_mtime = ""
        if cache_file.exists():
            cache_mtime = datetime.fromtimestamp(
                cache_file.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        all_venues.extend(venue_dicts)
        source_meta.append({
            "name": source.name,
            "description": source.description,
            "count": len(venue_dicts),
            "url": source.url,
            "from_cache": from_cache,
            "last_refreshed": cache_mtime,
        })

        if hasattr(source, "bin_dedup_count") and source.bin_dedup_count:
            pipeline_stats.append({
                "label": f"{source.name.upper()} BIN dedup (same name + building)",
                "removed": source.bin_dedup_count,
                "detail": f"{source.raw_camis_count} unique CAMIS → {source.raw_camis_count - source.bin_dedup_count} after dedup",
            })
        if hasattr(source, "dedup_count") and source.dedup_count:
            pipeline_stats.append({
                "label": f"{source.name.upper()} license dedup (same name + address)",
                "removed": source.dedup_count,
                "detail": f"{source.raw_count} raw → {source.raw_count - source.dedup_count} after dedup",
            })

    log.info("Total venues (pre-merge): %d from %d sources", len(all_venues), len(source_meta))

    # Cross-source merge
    merge_conf = app_conf.get("merge", {})
    if merge_conf.get("pairs"):
        all_venues, merge_stats = merge_cross_source(all_venues)
        log.info("Total venues (post-merge): %d", len(all_venues))
    else:
        merge_stats = {"pre_merge": len(all_venues), "pass1": 0, "pass2": 0, "pass3": 0,
                       "total_merges": 0, "post_merge": len(all_venues)}

    # Overrides
    overrides_file = app_dir / "overrides.json"
    applied = apply_overrides(all_venues, overrides_file)
    if applied:
        log.info("Overrides: applied %d", applied)

    # Borough bbox validation
    bad_coords = validate_coords(all_venues)
    if bad_coords:
        log.info("Dropped %d venues with coords outside borough bbox", bad_coords)

    # Cross-reference stamping
    xref_checked, coords_upgraded, xs = stamp_crossref(all_venues, app_dir)
    if xref_checked:
        log.info("Cross-ref: stamped %d venues (%d coords upgraded) | yelp=%s google=%s opentable=%s",
                 xref_checked, coords_upgraded, xs.get("yelp", {}), xs.get("google", {}), xs.get("opentable", {}))

    # Dietary tags (food-specific)
    diet_conf = app_conf.get("diet", {})
    diet_counts: dict[str, int] = {}
    diet_source_stats: dict[str, dict[str, int]] = {}
    hms_matched = 0
    knm_matched = 0
    if diet_conf.get("enabled"):
        diet_counts, diet_source_stats, hms_matched, knm_matched = apply_diet_tags(all_venues, app_dir, use_cache)
        all_diets = sorted(diet_counts.keys())
        log.info("Dietary tags: %s", {d: diet_counts[d] for d in all_diets})
        log.info("Authoritative matches: HMS=%d, KNM=%d", hms_matched, knm_matched)
    else:
        all_diets = []

    # Strip internal keys
    for v in all_venues:
        v.pop("yelp_cats", None)

    # Assign stable venue IDs for deep linking (e.g. eat.dash.nyc#joes-pizza-a3f2)
    seen_ids: set[str] = set()
    for v in all_venues:
        vid = make_venue_id(v.get("name", ""), v.get("address", ""), v.get("borough", ""))
        if vid in seen_ids:
            counter = 2
            while f"{vid}-{counter}" in seen_ids:
                counter += 1
            vid = f"{vid}-{counter}"
        seen_ids.add(vid)
        v["id"] = vid
    log.info("Assigned %d stable venue IDs", len(seen_ids))

    all_tags = sorted({t for v in all_venues for t in v.get("tags", [])})
    all_source_names = sorted({v["source"] for v in all_venues})

    # Compute per-venue filter tags (_ft) — single source of truth for all templates
    filter_list, filter_labels = _compute_venue_filters(all_venues, app_id, app_conf, all_diets)

    # --- Render ---
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap-only apps (vibe) skip venues.js / venue_info — only heatpoints.js
    heatmap_only = app_conf.get("heatmap_only", False)

    # Write venues.js
    if not heatmap_only:
        data_content = (
            f"// Generated {datetime.now(timezone.utc).isoformat()} git:{sha}\n"
            f"const VENUE_DATA = {json.dumps(all_venues, separators=(',', ':'))};\n"
            f"const SOURCE_META = {json.dumps(source_meta, separators=(',', ':'))};\n"
            f"const ALL_TAGS = {json.dumps(all_tags)};\n"
            f"const ALL_SOURCES = {json.dumps(all_source_names)};\n"
            f"const ALL_DIETS = {json.dumps(all_diets)};\n"
            f"const DIET_SOURCE_STATS = {json.dumps(diet_source_stats, separators=(',', ':'))};\n"
            f"const FILTERS = {json.dumps(filter_list)};\n"
            f"const FILTER_LABELS_MAP = {json.dumps(filter_labels, separators=(',', ':'))};\n"
        )
        data_js = dist_dir / "venues.js"
        data_js.write_text(data_content)
        _gzip_file(data_js)
        data_hash = hashlib.sha256(data_content.encode()).hexdigest()[:12]
        log.info("Wrote %s (%.1f MB, hash=%s)", data_js, data_js.stat().st_size / 1e6, data_hash)
    else:
        data_hash = ""

    # Write compact heatpoints.js for heatmap apps (vibe) — lat/lng/cat only
    if app_id == "vibe":
        _HEAT_SRC_TO_CAT = {
            "dohmh": 0, "sla": 1, "coffee": 1, "grocery": 2, "greenmarket": 2,
            "dpr": 0, "news": 3, "playgrounds": 6, "parks": 7,
            "cultural": 4, "shops": 5, "restrooms": 8,
            "theaters": 9, "cinemas": 10, "musicians": 11,
            "workshops": 12, "weather_stations": 13,
        }
        hp = []
        for v in all_venues:
            cat_idx = _HEAT_SRC_TO_CAT.get(v.get("source"))
            if cat_idx is not None and v.get("lat") is not None and v.get("lng") is not None:
                hp.append([round(v["lat"], 5), round(v["lng"], 5), cat_idx])
        hp_content = (
            f"// Compact heatmap data — generated {datetime.now(timezone.utc).isoformat()} git:{sha}\n"
            f"var HEAT_CATS=['eat','drink','food','news','art','shop','play','relax','pee','theater','movies','music','make','weather'];\n"
            f"var HEAT_POINTS={json.dumps(hp, separators=(',', ':'))};\n"
        )
        hp_js = dist_dir / "heatpoints.js"
        hp_js.write_text(hp_content)
        _gzip_file(hp_js)
        hp_hash = hashlib.sha256(hp_content.encode()).hexdigest()[:12]
        log.info("Wrote %s (%.1f MB → %.1f MB gz, hash=%s, %d points)", hp_js, hp_js.stat().st_size / 1e6, (hp_js.parent / (hp_js.name + '.gz')).stat().st_size / 1e6, hp_hash, len(hp))

    # Write venue_info.js (pre-rendered info panel HTML)
    if heatmap_only:
        venue_infos = []
    else:
        venue_infos = render_all_venue_infos(all_venues, app_id)
    if venue_infos:
        vi_content = (
            f"// Pre-rendered venue info panels — generated {datetime.now(timezone.utc).isoformat()} git:{sha}\n"
            f"const VENUE_INFO = {json.dumps(venue_infos, separators=(',', ':'))};\n"
        )
        vi_js = dist_dir / "venue_info.js"
        vi_js.write_text(vi_content)
        _gzip_file(vi_js)
        vi_hash = hashlib.sha256(vi_content.encode()).hexdigest()[:12]
        log.info("Wrote %s (%.1f MB → %.1f MB gz, hash=%s, %d panels)", vi_js, vi_js.stat().st_size / 1e6, (vi_js.parent / (vi_js.name + '.gz')).stat().st_size / 1e6, vi_hash, len(venue_infos))
    else:
        vi_hash = ""

    # Write venue_info.json (JSON array for SPA dynamic loading)
    if venue_infos:
        vi_json_content = json.dumps(venue_infos, separators=(",", ":"))
        vi_json_path = dist_dir / "venue_info.json"
        vi_json_path.write_text(vi_json_content)
        _gzip_file(vi_json_path)
        log.info("Wrote %s/venue_info.json (%.1f MB)", dist_dir, len(vi_json_content) / 1e6)

    # Write venues.json (for SPA/portal consumption via fetch())
    if not heatmap_only:
        json_payload = json.dumps({
            "venues": all_venues,
            "source_meta": source_meta,
            "tags": all_tags,
            "sources": all_source_names,
            "diets": all_diets,
            "diet_source_stats": diet_source_stats,
            "filters": filter_list,
            "filter_labels": filter_labels,
        }, separators=(",", ":"))
        vj_path = dist_dir / "venues.json"
        vj_path.write_text(json_payload)
        _gzip_file(vj_path)
        log.info("Wrote %s/venues.json (%.1f MB)", dist_dir, len(json_payload) / 1e6)

    # Write source_info.json (lightweight manifest for portal aggregation)
    source_info = {
        "app_id": app_id,
        "title": app_conf["title"],
        "built_at": datetime.now(timezone.utc).isoformat(),
        "venue_count": len(all_venues),
        "sources": source_meta,
    }
    (dist_dir / "source_info.json").write_text(json.dumps(source_info, separators=(",", ":")))
    log.info("Wrote %s/source_info.json", dist_dir)

    # Copy shared static assets first (as base layer)
    shared_static = ROOT / "shared" / "static"
    if shared_static.exists():
        for f in shared_static.iterdir():
            shutil.copy2(f, dist_dir / f.name)
            if f.name == "style.css":
                css_hash = hashlib.sha256(f.read_bytes()).hexdigest()[:12]

    # Copy app static assets (overrides shared)
    css_hash = ""
    static_dir = app_dir / "static"
    if static_dir.exists():
        for f in static_dir.iterdir():
            shutil.copy2(f, dist_dir / f.name)
            if f.name == "style.css":
                css_hash = hashlib.sha256(f.read_bytes()).hexdigest()[:12]

    # If no app-specific style.css, use the shared one's hash
    if not css_hash and shared_static.exists():
        shared_css = shared_static / "style.css"
        if shared_css.exists():
            css_hash = hashlib.sha256(shared_css.read_bytes()).hexdigest()[:12]

    # Copy shared TuiCss dist assets
    tuicss_dist = VENDOR / "TuiCss" / "dist"
    if tuicss_dist.exists():
        for name in ("tuicss.min.css", "tuicss.min.js"):
            src = tuicss_dist / name
            if src.exists():
                shutil.copy2(src, dist_dir / name)
        for subdir in ("fonts", "images"):
            src_dir = tuicss_dist / subdir
            dst_dir = dist_dir / subdir
            if src_dir.exists():
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)

    # Copy any app-level or shared static files (og-image, favicon, etc.)
    for extra in ("og-image.png", "og-image.svg", "favicon.svg"):
        src = app_dir / "static" / extra
        if not src.exists():
            src = shared_static / extra
        if src.exists():
            shutil.copy2(src, dist_dir / extra)

    # Render HTML from app-specific template
    template_dirs = [str(app_dir / "templates")]
    env = Environment(loader=FileSystemLoader(template_dirs), autoescape=True)
    template = env.get_template("index.html")
    # Load sub-app source info for portal apps (pre-render in template)
    sub_app_sources = []
    for sa in app_conf.get("sub_apps", []):
        sa_id = sa["id"]
        si_path = ROOT / "dist" / sa_id / "source_info.json"
        if si_path.exists():
            try:
                si_data = json.loads(si_path.read_text())
                sub_app_sources.append({"app": sa, "info": si_data})
            except Exception:
                log.warning("Failed to load source_info for sub-app %s", sa_id)
        else:
            log.debug("No source_info.json for sub-app %s (not yet built?)", sa_id)

    # -- Dash portal: inline all CSS/JS for zero-waterfall page load ------
    inline_assets: dict[str, str] = {}
    tile_bounds: dict = {}
    if app_id == "dash":
        _vendor_files = [
            ("leaflet_js",      ROOT / "vendor" / "leaflet" / "leaflet.js"),
            ("leaflet_css",     ROOT / "vendor" / "leaflet" / "leaflet.css"),
            ("supercluster_js", ROOT / "vendor" / "supercluster" / "supercluster.min.js"),
            ("tuicss_css",      VENDOR / "TuiCss" / "dist" / "tuicss.min.css"),
            ("tuicss_js",       VENDOR / "TuiCss" / "dist" / "tuicss.min.js"),
            ("fairy_js",        ROOT / "shared" / "static" / "fairy.js"),
            ("style_css",       ROOT / "shared" / "static" / "style.css"),
        ]
        for label, path in _vendor_files:
            if path.exists():
                inline_assets[label] = path.read_text()
            else:
                log.error("Missing asset for inlining: %s", path)
        log.info("Inlined %d assets (%.1f KB raw)",
                 len(inline_assets), sum(len(v) for v in inline_assets.values()) / 1024)

        # Pre-download map tiles for the default viewport (zoom 12, NYC)
        tile_bounds = _download_map_tiles(
            dist_dir / "tiles", ROOT / ".cache" / "tiles", zoom=12)

    html = template.render(
        app=app_conf,
        build_time=datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET"),
        build_sha=sha,
        venue_count=len(all_venues),
        sources=source_meta,
        all_tags=all_tags,
        all_sources=all_source_names,
        data_hash=data_hash,
        hp_hash=hp_hash if app_id == "vibe" else "",
        css_hash=css_hash,
        vi_hash=vi_hash,
        pipeline_stats=pipeline_stats,
        merge_stats=merge_stats,
        all_diets=all_diets,
        diet_counts=diet_counts,
        diet_source_stats=diet_source_stats,
        hms_matched=hms_matched,
        knm_matched=knm_matched,
        sub_app_sources=sub_app_sources,
        tile_bounds=json.dumps(tile_bounds),
        **inline_assets,
    )
    (dist_dir / "index.html").write_text(html)
    _gzip_file(dist_dir / "index.html")

    # Write build manifest for deploy/smoke tests
    manifest = {
        "app_id": app_id,
        "sha": sha,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "venue_count": len(all_venues),
        "sources": [s["name"] for s in source_meta],
        "data_hash": data_hash,
        "domains": app_conf.get("domains", []),
    }
    (dist_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ------------------------------------------------------------------
    # Post-build validation: every defined tag / source / diet must have
    # at least one venue.  Empty categories indicate a data or code issue.
    # ------------------------------------------------------------------
    _tag_counts: dict[str, int] = {}
    _src_counts: dict[str, int] = {}
    _diet_counts_check: dict[str, int] = {}
    for _v in all_venues:
        for _t in _v.get("tags") or []:
            _tag_counts[_t] = _tag_counts.get(_t, 0) + 1
        _src_counts[_v.get("source", "?")] = _src_counts.get(_v.get("source", "?"), 0) + 1
        for _d in _v.get("diet") or []:
            _diet_counts_check[_d] = _diet_counts_check.get(_d, 0) + 1

    _warnings: list[str] = []
    for _t in all_tags:
        if _tag_counts.get(_t, 0) == 0:
            _warnings.append(f"tag '{_t}' has 0 venues")
    for _s in all_source_names:
        if _src_counts.get(_s, 0) == 0:
            _warnings.append(f"source '{_s}' has 0 venues")
    for _d in all_diets:
        if _diet_counts_check.get(_d, 0) == 0:
            _warnings.append(f"diet '{_d}' has 0 venues")

    if _warnings:
        log.warning("DATA QUALITY — %d empty categories in %s:", len(_warnings), app_id)
        for _w in _warnings:
            log.warning("  ⚠ %s", _w)
    else:
        log.info("Validation OK — all %d tags, %d sources, %d diets have entries",
                 len(all_tags), len(all_source_names), len(all_diets))

    log.info("Build complete for %s → %s/ (%d venues, sha=%s)", app_id, dist_dir, len(all_venues), sha)
    return dist_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NYC Map — build static site(s)")
    parser.add_argument("apps", nargs="*", help="App IDs to build (default: all)")
    parser.add_argument("--all", action="store_true", help="Build all apps")
    parser.add_argument("--cache", action="store_true", help="Use cached data if < 24h old")
    parser.add_argument("--sources", type=str, default=None, help="Comma-separated source filter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine which apps to build
    if args.all:
        app_ids = list_apps()
    elif args.apps:
        app_ids = args.apps
    else:
        available = list_apps()
        if len(available) == 1:
            app_ids = available
        else:
            parser.error(f"Specify app(s) to build or use --all. Available: {', '.join(available)}")

    selected = set(args.sources.split(",")) if args.sources else None

    for app_id in app_ids:
        build_app(app_id, use_cache=args.cache, selected_sources=selected)

    # When building all apps, copy sub-app dist dirs into dash's dist so they
    # are served at dash.nyc/eat, dash.nyc/drink, dash.nyc/food.
    if args.all or "dash" in app_ids:
        dash_dist = ROOT / "dist" / "dash"
        if dash_dist.exists():
            for sub_id in ("eat", "drink", "food", "news", "art", "shop", "theater", "play", "movies", "music", "make", "relax", "pee", "weather", "today", "vibe"):
                sub_dist = ROOT / "dist" / sub_id
                if not sub_dist.exists():
                    log.warning("dist/%s not built — skipping copy into dash dist", sub_id)
                    continue
                target = dash_dist / sub_id
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(sub_dist, target)
                log.info("Copied dist/%s → dist/dash/%s/ for path-based routing", sub_id, sub_id)


if __name__ == "__main__":
    main()

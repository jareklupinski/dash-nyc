"""
NYC Public Restrooms — Directory of Toilets in Public Parks.

Source: https://data.cityofnewyork.us/Recreation/Directory-of-Toilets-in-Public-Parks/hjae-yuav
~616 restrooms in NYC parks.

Coordinates come from cross-referencing park names against the DPR Parks
Properties dataset (ghu2-eden) which has polygon geometry.  For any parks
not found in DPR, we fall back to Nominatim (OpenStreetMap) geocoding with
aggressive caching so the API is only hit once per unique address.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

RESTROOMS_DATASET = "hjae-yuav"
RESTROOMS_URL = f"https://data.cityofnewyork.us/resource/{RESTROOMS_DATASET}.json"

PARKS_DATASET = "ghu2-eden"
PARKS_URL = f"https://data.cityofnewyork.us/resource/{PARKS_DATASET}.json"

# NYC bounding box for coord sanity-check
_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)

# Borough name → single-letter code used in DPR Parks dataset
_BORO_CODE = {
    "BRONX": "X", "BROOKLYN": "B", "MANHATTAN": "M",
    "QUEENS": "Q", "STATEN ISLAND": "R",
}


def _centroid(geom: dict) -> tuple[float, float] | None:
    """Compute centroid from a GeoJSON Polygon or MultiPolygon."""
    coords = geom.get("coordinates", [])
    pts: list[list[float]] = []
    gtype = geom.get("type", "")
    if gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                pts.extend(ring)
    elif gtype == "Polygon":
        for ring in coords:
            pts.extend(ring)
    if not pts:
        return None
    lat = sum(p[1] for p in pts) / len(pts)
    lng = sum(p[0] for p in pts) / len(pts)
    return lat, lng


def _build_park_lookup() -> dict[str, tuple[float, float]]:
    """Fetch all DPR Parks Properties and build a name→centroid lookup."""
    log.info("Restrooms: loading DPR Parks Properties for coordinate lookup …")
    resp = requests.get(PARKS_URL, params={
        "$limit": 50000,
        "$select": "signname,name311,borough,the_geom",
    }, timeout=60)
    resp.raise_for_status()
    parks = resp.json()
    log.info("Restrooms: %d park polygons loaded", len(parks))

    lookup: dict[str, tuple[float, float]] = {}
    for p in parks:
        c = _centroid(p.get("the_geom", {}))
        if not c:
            continue
        for field in ("signname", "name311"):
            name = (p.get(field) or "").strip().upper()
            if not name:
                continue
            boro = (p.get("borough") or "").strip().upper()
            lookup[f"{name}|{boro}"] = c
            if name not in lookup:
                lookup[name] = c
    log.info("Restrooms: park lookup has %d keys", len(lookup))
    return lookup


def _match_park(name: str, boro_code: str, lookup: dict) -> tuple[float, float] | None:
    """Try multiple strategies to match a restroom name to a park centroid."""
    # 1. Exact match with borough
    key = f"{name}|{boro_code}"
    if key in lookup:
        return lookup[key]
    # 2. Exact match without borough
    if name in lookup:
        return lookup[name]
    # 3. Strip parenthetical sub-park name
    parent = re.sub(r"\s*\(.*\)\s*$", "", name).strip()
    if parent != name:
        key2 = f"{parent}|{boro_code}"
        if key2 in lookup:
            return lookup[key2]
        if parent in lookup:
            return lookup[parent]
    # 4. Strip "(PS NNN)" suffix specifically
    no_ps = re.sub(r"\s*\(PS\s+\d+\)\s*$", "", name).strip()
    if no_ps != name and no_ps != parent:
        key3 = f"{no_ps}|{boro_code}"
        if key3 in lookup:
            return lookup[key3]
        if no_ps in lookup:
            return lookup[no_ps]
    # 5. Add common suffixes
    for suffix in (" PARK", " PLAYGROUND", " GARDEN"):
        alt = name + suffix
        key4 = f"{alt}|{boro_code}"
        if key4 in lookup:
            return lookup[key4]
        if alt in lookup:
            return lookup[alt]
    # 6. Normalize punctuation
    normalized = name.replace(".", "").replace("'S ", " ").replace("'S", "")
    if normalized != name:
        key5 = f"{normalized}|{boro_code}"
        if key5 in lookup:
            return lookup[key5]
    return None


def _nominatim_geocode(query: str, cache: dict[str, Any]) -> tuple[float, float] | None:
    """Geocode via Nominatim (free, 1 req/sec), with local JSON cache."""
    if query in cache:
        hit = cache[query]
        if hit:
            return hit["lat"], hit["lng"]
        return None

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "viewbox": "-74.27,40.92,-73.68,40.49",
                "bounded": 1,
            },
            headers={"User-Agent": "dash.nyc (mail@dash.nyc)"},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lng = float(results[0]["lon"])
            cache[query] = {"lat": lat, "lng": lng}
            return lat, lng
        else:
            cache[query] = None
            return None
    except Exception as exc:
        log.warning("Nominatim geocode failed for %r: %s", query, exc)
        return None
    finally:
        time.sleep(1.1)  # Nominatim rate limit: 1 req/sec


def _in_nyc(lat: float, lng: float) -> bool:
    lat_lo, lat_hi, lng_lo, lng_hi = _NYC_BBOX
    return lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi


class RestroomSource(DataSource):
    raw_count: int = 0
    park_matched: int = 0
    nominatim_matched: int = 0
    skipped: int = 0

    @property
    def name(self) -> str:
        return "restrooms"

    @property
    def description(self) -> str:
        return "NYC Parks Public Restrooms"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/Recreation/Directory-of-Toilets-in-Public-Parks/hjae-yuav"

    def fetch(self) -> list[Venue]:
        # 1. Fetch restroom directory
        log.info("Restrooms: fetching from NYC Open Data …")
        resp = requests.get(RESTROOMS_URL, params={"$limit": 5000}, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        log.info("Restrooms: %d rows from API", len(rows))

        # 2. Build park name → centroid lookup from DPR Parks Properties
        park_lookup = _build_park_lookup()

        # 3. Load Nominatim geocode cache (for fallback)
        geocode_cache_path = self.cache_dir / "restrooms_nominatim.json"
        nom_cache: dict[str, Any] = {}
        if geocode_cache_path.exists():
            try:
                nom_cache = json.loads(geocode_cache_path.read_text())
            except Exception:
                pass

        venues: list[Venue] = []
        nom_api_calls = 0

        for row in rows:
            self.raw_count += 1
            park_name = (row.get("name") or "").strip()
            location = (row.get("location") or "").strip()
            borough = (row.get("borough") or "").strip().upper()
            accessible = (row.get("handicap_accessible") or "").strip()
            year_round = (row.get("open_year_round") or "").strip()
            comments = (row.get("comments") or "").strip()

            if not park_name:
                self.skipped += 1
                continue

            boro_code = _BORO_CODE.get(borough, borough)
            name_upper = park_name.upper()

            # Strategy A: DPR Parks cross-reference
            coords = _match_park(name_upper, boro_code, park_lookup)
            if coords:
                self.park_matched += 1
            else:
                # Strategy B: Nominatim fallback
                boro_full = borough.title()
                query = f"{park_name}, {location}, {boro_full}, New York, NY" if location else f"{park_name}, {boro_full}, New York, NY"
                was_cached = query in nom_cache
                coords = _nominatim_geocode(query, nom_cache)
                if not was_cached and coords is not None:
                    nom_api_calls += 1
                if coords:
                    self.nominatim_matched += 1
                else:
                    self.skipped += 1
                    continue

            lat, lng = coords
            if not _in_nyc(lat, lng):
                self.skipped += 1
                continue

            tags: list[str] = ["restroom"]
            if accessible.lower() == "yes":
                tags.append("accessible")
            if year_round.lower() == "yes":
                tags.append("year_round")
            elif year_round.lower() == "no":
                tags.append("seasonal")

            meta: dict[str, Any] = {}
            if accessible:
                meta["accessible"] = accessible.lower() == "yes"
            if year_round:
                meta["year_round"] = year_round.lower() == "yes"
            if comments:
                meta["comments"] = comments

            venues.append(Venue(
                name=park_name,
                lat=lat,
                lng=lng,
                source="restrooms",
                address=location,
                borough=borough,
                tags=tags,
                meta=meta,
            ))

        # Save Nominatim cache
        geocode_cache_path.parent.mkdir(parents=True, exist_ok=True)
        geocode_cache_path.write_text(json.dumps(nom_cache, indent=2))

        log.info(
            "Restrooms: %d venues | park_match=%d, nominatim=%d (api=%d), skipped=%d",
            len(venues), self.park_matched, self.nominatim_matched,
            nom_api_calls, self.skipped,
        )
        return venues

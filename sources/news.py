"""
NYC News — Good-news RSS feed articles geocoded to NYC locations.

Reads pre-fetched article data from the good-news project and places
articles on the map by matching NYC place names (neighborhoods, landmarks,
parks, venues, sports teams, transit hubs) in their titles and summaries.

Articles that don't mention a recognizable NYC location are skipped.
Each article becomes a "venue" with sentiment score, category, and a link.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

# Path to the good-news project's output data — set via env or default
_DEFAULT_DATA = Path.home() / "good-news" / "public" / "data.json"

# ---------------------------------------------------------------------------
# NYC location dictionary: term → (lat, lng, borough, specificity)
#   specificity: higher = more specific (prefer over broader matches)
#     3 = exact landmark/venue
#     2 = neighborhood
#     1 = borough / broad area
# ---------------------------------------------------------------------------
_NYC_LOCATIONS: dict[str, tuple[float, float, str, int]] = {
    # --- Boroughs (broad) ---
    "manhattan":        (40.7831, -73.9712, "MANHATTAN", 1),
    "brooklyn":         (40.6782, -73.9442, "BROOKLYN", 1),
    "queens":           (40.7282, -73.7949, "QUEENS", 1),
    "the bronx":        (40.8448, -73.8648, "BRONX", 1),
    "bronx":            (40.8448, -73.8648, "BRONX", 1),
    "staten island":    (40.5795, -74.1502, "STATEN ISLAND", 1),

    # --- Broad terms ---
    "new york city":    (40.7128, -74.0060, "MANHATTAN", 1),
    "new york":         (40.7128, -74.0060, "MANHATTAN", 1),
    "nyc":              (40.7128, -74.0060, "MANHATTAN", 1),

    # --- Manhattan neighborhoods ---
    "lower manhattan":  (40.7075, -74.0113, "MANHATTAN", 2),
    "financial district": (40.7075, -74.0089, "MANHATTAN", 2),
    "fidi":             (40.7075, -74.0089, "MANHATTAN", 2),
    "tribeca":          (40.7163, -74.0086, "MANHATTAN", 2),
    "soho":             (40.7233, -73.9985, "MANHATTAN", 2),
    "noho":             (40.7264, -73.9927, "MANHATTAN", 2),
    "nolita":           (40.7234, -73.9955, "MANHATTAN", 2),
    "chinatown":        (40.7158, -73.9970, "MANHATTAN", 2),
    "little italy":     (40.7191, -73.9973, "MANHATTAN", 2),
    "lower east side":  (40.7150, -73.9843, "MANHATTAN", 2),
    "east village":     (40.7265, -73.9815, "MANHATTAN", 2),
    "west village":     (40.7336, -74.0027, "MANHATTAN", 2),
    "greenwich village": (40.7336, -74.0027, "MANHATTAN", 2),
    "chelsea":          (40.7465, -74.0014, "MANHATTAN", 2),
    "flatiron":         (40.7411, -73.9897, "MANHATTAN", 2),
    "gramercy":         (40.7368, -73.9845, "MANHATTAN", 2),
    "union square":     (40.7359, -73.9911, "MANHATTAN", 2),
    "murray hill":      (40.7489, -73.9759, "MANHATTAN", 2),
    "kips bay":         (40.7425, -73.9801, "MANHATTAN", 2),
    "midtown":          (40.7549, -73.9840, "MANHATTAN", 2),
    "hells kitchen":    (40.7638, -73.9918, "MANHATTAN", 2),
    "hell's kitchen":   (40.7638, -73.9918, "MANHATTAN", 2),
    "times square":     (40.7580, -73.9855, "MANHATTAN", 3),
    "theater district":  (40.7590, -73.9845, "MANHATTAN", 2),
    "garment district":  (40.7537, -73.9904, "MANHATTAN", 2),
    "upper west side":  (40.7870, -73.9754, "MANHATTAN", 2),
    "upper west":       (40.7870, -73.9754, "MANHATTAN", 2),
    "upper east side":  (40.7736, -73.9566, "MANHATTAN", 2),
    "upper east":       (40.7736, -73.9566, "MANHATTAN", 2),
    "harlem":           (40.8116, -73.9465, "MANHATTAN", 2),
    "east harlem":      (40.7957, -73.9425, "MANHATTAN", 2),
    "el barrio":        (40.7957, -73.9425, "MANHATTAN", 2),
    "spanish harlem":   (40.7957, -73.9425, "MANHATTAN", 2),
    "morningside heights": (40.8098, -73.9615, "MANHATTAN", 2),
    "morningside":      (40.8098, -73.9615, "MANHATTAN", 2),
    "washington heights": (40.8417, -73.9394, "MANHATTAN", 2),
    "inwood":           (40.8677, -73.9212, "MANHATTAN", 2),
    "marble hill":      (40.8764, -73.9107, "MANHATTAN", 2),

    # --- Brooklyn neighborhoods ---
    "williamsburg":     (40.7081, -73.9571, "BROOKLYN", 2),
    "greenpoint":       (40.7274, -73.9514, "BROOKLYN", 2),
    "dumbo":            (40.7033, -73.9883, "BROOKLYN", 2),
    "brooklyn heights": (40.6960, -73.9936, "BROOKLYN", 2),
    "downtown brooklyn": (40.6936, -73.9857, "BROOKLYN", 2),
    "park slope":       (40.6710, -73.9814, "BROOKLYN", 2),
    "prospect heights": (40.6779, -73.9690, "BROOKLYN", 2),
    "bed-stuy":         (40.6872, -73.9418, "BROOKLYN", 2),
    "bedford-stuyvesant": (40.6872, -73.9418, "BROOKLYN", 2),
    "bushwick":         (40.6944, -73.9213, "BROOKLYN", 2),
    "crown heights":    (40.6694, -73.9422, "BROOKLYN", 2),
    "flatbush":         (40.6521, -73.9590, "BROOKLYN", 2),
    "red hook":         (40.6734, -74.0081, "BROOKLYN", 2),
    "sunset park":      (40.6466, -74.0048, "BROOKLYN", 2),
    "bay ridge":        (40.6350, -74.0285, "BROOKLYN", 2),
    "cobble hill":      (40.6860, -73.9963, "BROOKLYN", 2),
    "carroll gardens":  (40.6802, -73.9993, "BROOKLYN", 2),
    "gowanus":          (40.6737, -73.9897, "BROOKLYN", 2),
    "borough park":     (40.6346, -73.9908, "BROOKLYN", 2),
    "bensonhurst":      (40.6023, -73.9937, "BROOKLYN", 2),
    "coney island":     (40.5749, -73.9859, "BROOKLYN", 2),
    "brighton beach":   (40.5776, -73.9597, "BROOKLYN", 2),
    "sheepshead bay":   (40.5903, -73.9443, "BROOKLYN", 2),
    "fort greene":      (40.6892, -73.9764, "BROOKLYN", 2),
    "boerum hill":      (40.6847, -73.9835, "BROOKLYN", 2),
    "clinton hill":     (40.6890, -73.9662, "BROOKLYN", 2),

    # --- Queens neighborhoods ---
    "astoria":          (40.7724, -73.9301, "QUEENS", 2),
    "long island city":  (40.7447, -73.9485, "QUEENS", 2),
    "lic":              (40.7447, -73.9485, "QUEENS", 2),
    "jackson heights":  (40.7557, -73.8831, "QUEENS", 2),
    "flushing":         (40.7675, -73.8330, "QUEENS", 2),
    "forest hills":     (40.7182, -73.8448, "QUEENS", 2),
    "jamaica":          (40.7028, -73.7890, "QUEENS", 2),
    "woodside":         (40.7456, -73.9026, "QUEENS", 2),
    "sunnyside":        (40.7434, -73.9196, "QUEENS", 2),
    "corona":           (40.7450, -73.8623, "QUEENS", 2),
    "rockaway":         (40.5860, -73.8135, "QUEENS", 2),
    "rockaway beach":   (40.5834, -73.8161, "QUEENS", 2),
    "bayside":          (40.7627, -73.7716, "QUEENS", 2),
    "ridgewood":        (40.7043, -73.9055, "QUEENS", 2),
    "elmhurst":         (40.7379, -73.8801, "QUEENS", 2),

    # --- Bronx neighborhoods ---
    "south bronx":      (40.8176, -73.9209, "BRONX", 2),
    "mott haven":       (40.8094, -73.9229, "BRONX", 2),
    "hunts point":      (40.8122, -73.8915, "BRONX", 2),
    "fordham":          (40.8615, -73.8918, "BRONX", 2),
    "pelham bay":       (40.8529, -73.8371, "BRONX", 2),
    "riverdale":        (40.8990, -73.9088, "BRONX", 2),
    "kingsbridge":      (40.8819, -73.8979, "BRONX", 2),
    "city island":      (40.8468, -73.7868, "BRONX", 2),
    "throgs neck":      (40.8260, -73.8198, "BRONX", 2),
    "concourse":        (40.8268, -73.9232, "BRONX", 2),

    # --- Landmarks & venues (high specificity) ---
    "central park":     (40.7829, -73.9654, "MANHATTAN", 3),
    "prospect park":    (40.6602, -73.9690, "BROOKLYN", 3),
    "brooklyn bridge":  (40.7061, -73.9969, "MANHATTAN", 3),
    "george washington bridge": (40.8517, -73.9527, "MANHATTAN", 3),
    "statue of liberty": (40.6892, -74.0445, "MANHATTAN", 3),
    "liberty island":   (40.6892, -74.0445, "MANHATTAN", 3),
    "ellis island":     (40.6995, -74.0396, "MANHATTAN", 3),
    "governors island": (40.6892, -74.0167, "MANHATTAN", 3),
    "empire state":     (40.7484, -73.9857, "MANHATTAN", 3),
    "chrysler building": (40.7516, -73.9755, "MANHATTAN", 3),
    "rockefeller center": (40.7587, -73.9787, "MANHATTAN", 3),
    "rockefeller":      (40.7587, -73.9787, "MANHATTAN", 3),
    "hudson yards":     (40.7534, -74.0010, "MANHATTAN", 3),
    "world trade center": (40.7127, -74.0134, "MANHATTAN", 3),
    "one world trade":  (40.7127, -74.0134, "MANHATTAN", 3),
    "wall street":      (40.7068, -74.0090, "MANHATTAN", 3),

    # --- Transit ---
    "grand central":    (40.7527, -73.9772, "MANHATTAN", 3),
    "penn station":     (40.7506, -73.9935, "MANHATTAN", 3),
    "jfk":              (40.6413, -73.7781, "QUEENS", 3),
    "laguardia":        (40.7769, -73.8740, "QUEENS", 3),
    "newark airport":   (40.6895, -74.1745, "MANHATTAN", 3),
    "port authority":   (40.7569, -73.9903, "MANHATTAN", 3),

    # --- Cultural venues ---
    "lincoln center":   (40.7725, -73.9835, "MANHATTAN", 3),
    "carnegie hall":    (40.7651, -73.9800, "MANHATTAN", 3),
    "broadway":         (40.7590, -73.9845, "MANHATTAN", 2),
    "met museum":       (40.7794, -73.9632, "MANHATTAN", 3),
    "metropolitan museum": (40.7794, -73.9632, "MANHATTAN", 3),
    "the met":          (40.7794, -73.9632, "MANHATTAN", 3),
    "guggenheim":       (40.7830, -73.9590, "MANHATTAN", 3),
    "moma":             (40.7614, -73.9776, "MANHATTAN", 3),
    "museum of modern art": (40.7614, -73.9776, "MANHATTAN", 3),
    "whitney museum":   (40.7396, -74.0089, "MANHATTAN", 3),
    "whitney":          (40.7396, -74.0089, "MANHATTAN", 3),
    "brooklyn museum":  (40.6712, -73.9636, "BROOKLYN", 3),
    "new museum":       (40.7224, -73.9929, "MANHATTAN", 3),
    "the cloisters":    (40.8649, -73.9318, "MANHATTAN", 3),
    "apollo theater":   (40.8100, -73.9500, "MANHATTAN", 3),
    "radio city":       (40.7600, -73.9799, "MANHATTAN", 3),
    "beacon theatre":   (40.7805, -73.9812, "MANHATTAN", 3),
    "city hall":        (40.7128, -74.0060, "MANHATTAN", 3),

    # --- Sports ---
    "madison square garden": (40.7505, -73.9934, "MANHATTAN", 3),
    "madison square":   (40.7505, -73.9934, "MANHATTAN", 3),
    "msg":              (40.7505, -73.9934, "MANHATTAN", 3),
    "barclays center":  (40.6826, -73.9754, "BROOKLYN", 3),
    "barclays":         (40.6826, -73.9754, "BROOKLYN", 3),
    "yankee stadium":   (40.8296, -73.9262, "BRONX", 3),
    "citi field":       (40.7571, -73.8458, "QUEENS", 3),
    "arthur ashe":      (40.7508, -73.8463, "QUEENS", 3),
    "usta":             (40.7508, -73.8463, "QUEENS", 3),
    "belmont park":     (40.7176, -73.7225, "QUEENS", 3),

    # --- Sports teams (map to their home venue) ---
    "yankees":          (40.8296, -73.9262, "BRONX", 3),
    "mets":             (40.7571, -73.8458, "QUEENS", 3),
    "knicks":           (40.7505, -73.9934, "MANHATTAN", 3),
    "rangers":          (40.7505, -73.9934, "MANHATTAN", 3),
    "nets":             (40.6826, -73.9754, "BROOKLYN", 3),
    "islanders":        (40.6868, -73.9753, "BROOKLYN", 3),
    "liberty":          (40.6826, -73.9754, "BROOKLYN", 3),
    "nycfc":            (40.8296, -73.9262, "BRONX", 3),
    "red bulls":        (40.7368, -74.1503, "MANHATTAN", 3),
    "giants":           (40.8128, -74.0742, "MANHATTAN", 3),
    "jets":             (40.8128, -74.0742, "MANHATTAN", 3),

    # --- Universities ---
    "columbia university": (40.8075, -73.9626, "MANHATTAN", 3),
    "columbia":         (40.8075, -73.9626, "MANHATTAN", 2),
    "nyu":              (40.7295, -73.9965, "MANHATTAN", 3),
    "new york university": (40.7295, -73.9965, "MANHATTAN", 3),
    "cuny":             (40.7484, -73.9838, "MANHATTAN", 2),
    "fordham university": (40.8614, -73.8855, "BRONX", 3),
    "hunter college":   (40.7685, -73.9656, "MANHATTAN", 3),
    "baruch":           (40.7404, -73.9831, "MANHATTAN", 3),
    "cooper union":     (40.7296, -73.9905, "MANHATTAN", 3),
    "pratt":            (40.6895, -73.9635, "BROOKLYN", 3),
    "parsons":          (40.7353, -73.9942, "MANHATTAN", 3),

    # --- Parks & recreation ---
    "high line":        (40.7480, -74.0048, "MANHATTAN", 3),
    "the highline":     (40.7480, -74.0048, "MANHATTAN", 3),
    "bryant park":      (40.7536, -73.9832, "MANHATTAN", 3),
    "washington square": (40.7308, -73.9975, "MANHATTAN", 3),
    "tompkins square":  (40.7265, -73.9817, "MANHATTAN", 3),
    "battery park":     (40.7033, -74.0170, "MANHATTAN", 3),
    "riverside park":   (40.8032, -73.9708, "MANHATTAN", 3),
    "van cortlandt park": (40.8972, -73.8986, "BRONX", 3),
    "pelham bay park":  (40.8671, -73.8100, "BRONX", 3),
    "flushing meadows":  (40.7400, -73.8408, "QUEENS", 3),
    "randalls island":  (40.7934, -73.9216, "MANHATTAN", 3),
    "roosevelt island":  (40.7620, -73.9510, "MANHATTAN", 3),

    # --- Food & drink landmarks ---
    "smorgasburg":      (40.7215, -73.9612, "BROOKLYN", 3),
    "chelsea market":   (40.7424, -74.0061, "MANHATTAN", 3),
    "eataly":           (40.7422, -73.9896, "MANHATTAN", 3),

}

# Pre-compile word-boundary patterns for every location term.
# Using word boundaries for ALL terms prevents substring false positives
# (e.g. "lic" in "public", "usta" in "sustained", etc.)
_LOCATION_PATTERNS: dict[str, re.Pattern[str]] = {
    term: re.compile(r'\b' + re.escape(term) + r'\b')
    for term in _NYC_LOCATIONS
}

# ---------------------------------------------------------------------------
# Negative-keyword override: reclassify "POSITIVE" articles that contain
# obviously negative phrases the upstream classifier missed.
# ---------------------------------------------------------------------------
_NEGATIVE_KEYWORDS: list[str] = [
    # death / violence
    "dies at", "died", "death of", "killed", "murder", "shooting",
    "shot by", "gunfire", "stabbed", "fatal",
    # crime / justice
    "charged with", "charges", "arrested", "indicted", "convicted",
    "assault", "robbery", "theft", "fraud", "scandal",
    # fear / conflict
    "fearing", "feared", "warships", "fighter jets", "military buildup",
    "missile", "airstrike", "bomb", "explosion",
    # disasters
    "wildfire", "earthquake", "flood", "collapsed", "derailed",
]
_NEG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in _NEGATIVE_KEYWORDS
]


def _override_sentiment(label: str, title: str, summary: str) -> str:
    """Reclassify obviously negative articles the upstream model mislabelled."""
    if label != "POSITIVE":
        return label
    text = f"{title} {summary}"
    for pat in _NEG_PATTERNS:
        if pat.search(text):
            return "NEGATIVE"
    return label


def _best_location(text: str) -> tuple[str, float, float, str] | None:
    """Find the most specific NYC location mentioned in text.

    Returns (matched_term, lat, lng, borough) or None.
    """
    text_lower = text.lower()
    best: tuple[str, float, float, str, int] | None = None

    for term, (lat, lng, boro, spec) in _NYC_LOCATIONS.items():
        if not _LOCATION_PATTERNS[term].search(text_lower):
            continue

        # Pick the most specific match (highest specificity, then longest term)
        if best is None or spec > best[4] or (spec == best[4] and len(term) > len(best[0])):
            best = (term, lat, lng, boro, spec)

    if best:
        return (best[0], best[1], best[2], best[3])
    return None


class NewsSource(DataSource):
    """NYC news articles geocoded from good-news RSS feeds."""

    @property
    def name(self) -> str:
        return "news"

    @property
    def description(self) -> str:
        return "NYC News — RSS feeds geocoded to city locations"

    @property
    def url(self) -> str:
        return ""

    def fetch(self) -> list[Venue]:
        import os
        data_path = Path(os.environ.get("GOOD_NEWS_DATA", str(_DEFAULT_DATA)))

        if not data_path.exists():
            log.warning("News data not found at %s — run good-news fetch first", data_path)
            return []

        data = json.loads(data_path.read_text())
        articles = data.get("articles", [])
        log.info("News: loaded %d articles from %s", len(articles), data_path)

        venues: list[Venue] = []
        skipped = 0

        for article in articles:
            title = article.get("title", "")
            summary = article.get("summary", "")
            text = f"{title} {summary}"

            loc = _best_location(text)
            if not loc:
                skipped += 1
                continue

            matched_term, lat, lng, borough = loc

            # Add slight jitter so overlapping articles don't stack exactly
            import hashlib
            h = hashlib.md5(title.encode()).hexdigest()
            jitter_lat = (int(h[:4], 16) - 0x8000) / 0x8000 * 0.003
            jitter_lng = (int(h[4:8], 16) - 0x8000) / 0x8000 * 0.003
            lat += jitter_lat
            lng += jitter_lng

            label = article.get("label", "POSITIVE")
            label = _override_sentiment(label, title, summary)
            score = article.get("score", 0.5)
            category = article.get("category", "news")
            source_name = article.get("source", "")

            tags = [category, label.lower()]

            meta: dict[str, Any] = {
                "link": article.get("link", ""),
                "summary": summary[:200] if summary else "",
                "published": article.get("published", ""),
                "feed": source_name,
                "sentiment": label,
                "score": round(score, 4),
                "matched_location": matched_term,
            }

            venues.append(Venue(
                name=title,
                lat=lat,
                lng=lng,
                source="news",
                address=matched_term.title(),
                borough=borough,
                cuisine=category,
                tags=tags,
                meta=meta,
            ))

        log.info(
            "News: %d geocoded articles, %d skipped (no NYC location)",
            len(venues), skipped,
        )
        pos = sum(1 for v in venues if "positive" in v.tags)
        log.info("News: %d positive, %d negative", pos, len(venues) - pos)
        return venues

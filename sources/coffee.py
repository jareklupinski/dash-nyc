"""
Coffee Shops, Tea Houses & Juice Bars — filtered from DOHMH inspection data.

Only includes venues whose cuisine_description matches coffee, tea,
juice, smoothie, or café categories.  Used by drink.dash.nyc.

Source: https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

DATASET_ID = "43nn-pn8j"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
PAGE_SIZE = 50_000

# Cuisine descriptions that indicate coffee / tea / juice venues
_DRINK_CUISINES = {
    "café/coffee/tea",
    "coffee/tea",
    "juice, smoothies, fruit salads",
    "bottled beverages, water, natural juices",
    "ice cream, gelato, yogurt, ices",
}

# Partial matches (substring in cuisine_description.lower())
_DRINK_PARTIALS = ("coffee", "tea house", "juice", "smoothie", "bubble tea", "boba")


def _is_drink_venue(cuisine: str) -> bool:
    """Return True if the cuisine indicates a coffee/tea/juice venue."""
    cl = cuisine.strip().lower()
    if cl in _DRINK_CUISINES:
        return True
    return any(p in cl for p in _DRINK_PARTIALS)


class CoffeeSource(DataSource):
    raw_camis_count: int = 0
    bin_dedup_count: int = 0

    @property
    def name(self) -> str:
        return "coffee"

    @property
    def description(self) -> str:
        return "NYC Coffee Shops, Tea Houses & Juice Bars (DOHMH)"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j"

    def fetch(self) -> list[Venue]:
        """Fetch coffee/tea/juice venues from DOHMH inspection data."""
        params: dict[str, Any] = {
            "$select": (
                "camis,dba,building,street,boro,zipcode,phone,"
                "cuisine_description,grade,bin,latitude,longitude,inspection_date"
            ),
            "$where": "latitude IS NOT NULL AND longitude IS NOT NULL",
            "$order": "camis",
            "$limit": PAGE_SIZE,
        }

        token = os.environ.get("NYC_OPEN_DATA_TOKEN")
        headers = {}
        if token:
            headers["X-App-Token"] = token

        seen: dict[str, Venue] = {}
        offset = 0

        while True:
            params["$offset"] = offset
            log.info("Coffee: fetching offset %d …", offset)

            resp = requests.get(BASE_URL, params=params, headers=headers, timeout=120)
            resp.raise_for_status()
            rows = resp.json()

            if not rows:
                break

            for row in rows:
                camis = row.get("camis", "")
                if not camis or camis in seen:
                    continue

                cuisine = row.get("cuisine_description", "")
                if not _is_drink_venue(cuisine):
                    continue

                try:
                    lat = float(row.get("latitude", 0))
                    lng = float(row.get("longitude", 0))
                except (TypeError, ValueError):
                    continue

                if lat == 0 or lng == 0:
                    continue

                insp_date = row.get("inspection_date", "")
                if insp_date:
                    insp_date = insp_date[:10]

                address_parts = [
                    row.get("building", ""),
                    row.get("street", ""),
                ]
                address = " ".join(p for p in address_parts if p).strip()

                # Determine specific drink tags
                cl = cuisine.strip().lower()
                tags = ["drink"]
                if "coffee" in cl or "café" in cl or "cafe" in cl or "tea" in cl:
                    tags.append("coffee_tea")
                if "juice" in cl or "smoothie" in cl:
                    tags.append("juice")
                if "ice cream" in cl or "gelato" in cl or "yogurt" in cl:
                    tags.append("frozen")
                if "bubble" in cl or "boba" in cl:
                    tags.append("bubble_tea")

                seen[camis] = Venue(
                    name=row.get("dba", "Unknown").title(),
                    lat=lat,
                    lng=lng,
                    source=self.name,
                    address=address,
                    cuisine=cuisine,
                    borough=row.get("boro", ""),
                    phone=row.get("phone", ""),
                    grade=row.get("grade", ""),
                    zipcode=row.get("zipcode", ""),
                    tags=tags,
                    meta={"bin": row.get("bin", "")},
                    opened=insp_date,
                )

            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break

        # Dedup by normalized name + BIN
        by_name_bin: dict[tuple[str, str], Venue] = {}
        for v in seen.values():
            bldg_id = v.meta.get("bin", "")
            norm_name = re.sub(r"[^a-z0-9 ]", "", v.name.lower()).strip()
            norm_name = re.sub(r"\s+", " ", norm_name)
            key = (norm_name, bldg_id) if bldg_id else (norm_name, str(id(v)))
            if key in by_name_bin:
                existing = by_name_bin[key]
                if v.grade and not existing.grade:
                    by_name_bin[key] = v
                elif v.cuisine and not existing.cuisine:
                    by_name_bin[key] = v
            else:
                by_name_bin[key] = v

        venues = list(by_name_bin.values())
        deduped = len(seen) - len(venues)
        self.raw_camis_count = len(seen)
        self.bin_dedup_count = deduped
        if deduped:
            log.info("Coffee: deduped %d same-name-same-building entries", deduped)
        log.info("Coffee: fetched %d coffee/tea/juice venues", len(venues))
        return venues

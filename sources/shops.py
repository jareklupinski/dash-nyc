"""
NYC Licensed Retail & Shopping — from DCWP Licensed Businesses.

Source: https://data.cityofnewyork.us/Business/Legally-Operating-Businesses/w7w3-xahh
Includes secondhand dealers (thrift/vintage), electronics stores,
general vendors (street vendors), stoop line stands, and newsstands.
~8k active businesses across all five boroughs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

DATASET_ID = "w7w3-xahh"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"

# Business categories relevant to "shopping"
_SHOP_CATEGORIES = (
    "Secondhand Dealer - General",
    "Electronics Store",
    "General Vendor",
    "Stoop Line Stand",
    "Newsstand",
    "Pawnbroker",
)

# Tag mapping
_CATEGORY_TAGS: dict[str, str] = {
    "Secondhand Dealer - General": "thrift",
    "Electronics Store": "electronics",
    "General Vendor": "vendor",
    "Stoop Line Stand": "sidewalk",
    "Newsstand": "newsstand",
    "Pawnbroker": "pawnshop",
}

# NYC bounding box
_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)


class ShopSource(DataSource):
    raw_count: int = 0
    bad_coords: int = 0

    @property
    def name(self) -> str:
        return "shops"

    @property
    def description(self) -> str:
        return "NYC Licensed Retail & Shopping Businesses"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/Business/Legally-Operating-Businesses/w7w3-xahh"

    def fetch(self) -> list[Venue]:
        token = os.environ.get("NYC_OPEN_DATA_TOKEN", "")
        cat_list = ",".join(f"'{c}'" for c in _SHOP_CATEGORIES)
        params: dict[str, Any] = {
            "$limit": 50000,
            "$where": (
                f"business_category in({cat_list}) "
                "AND license_status = 'Active' "
                "AND latitude IS NOT NULL "
                "AND longitude IS NOT NULL"
            ),
        }
        if token:
            params["$$app_token"] = token

        log.info("Shops: fetching from DCWP Licensed Businesses …")
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        log.info("Shops: %d rows from API", len(rows))

        venues: list[Venue] = []
        for row in rows:
            self.raw_count += 1
            v = self._venue_from_row(row)
            if v:
                venues.append(v)
            else:
                self.bad_coords += 1

        log.info(
            "Shops: %d venues (from %d rows, %d bad coords)",
            len(venues), self.raw_count, self.bad_coords,
        )
        return venues

    @staticmethod
    def _venue_from_row(row: dict[str, Any]) -> Venue | None:
        biz_name = (row.get("business_name") or "").strip()
        if not biz_name:
            return None

        try:
            lat = float(row["latitude"])
            lng = float(row["longitude"])
        except (KeyError, ValueError, TypeError):
            return None

        lat_lo, lat_hi, lng_lo, lng_hi = _NYC_BBOX
        if not (lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi):
            return None

        borough = (row.get("address_borough") or "").strip().upper()
        building = (row.get("address_building") or "").strip()
        street = (row.get("address_street_name") or "").strip()
        address = f"{building} {street}".strip() if building or street else ""
        zipcode = (row.get("address_zip") or "").strip()
        phone = (row.get("contact_phone") or "").strip()
        category = (row.get("business_category") or "").strip()

        tags: list[str] = []
        tag = _CATEGORY_TAGS.get(category, "retail")
        tags.append(tag)

        meta: dict[str, Any] = {}
        if category:
            meta["license_type"] = category

        return Venue(
            name=biz_name.title(),
            lat=lat,
            lng=lng,
            source="shops",
            address=address,
            borough=borough,
            zipcode=zipcode,
            phone=phone,
            cuisine=category,
            tags=tags,
            meta=meta,
        )

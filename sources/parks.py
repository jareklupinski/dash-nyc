"""
NYC Parks, Gardens & Green Spaces — from the City Planning Facilities Database.

Source: https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5
Includes neighborhood parks, community parks, flagship parks, nature areas,
gardens, state parks, and other serene green spaces.  ~900 venues across
all five boroughs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

DATASET_ID = "ji82-xba5"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"

# Facility types that map to "relaxation" green spaces
_PARK_TYPES = (
    "NEIGHBORHOOD PARK",
    "COMMUNITY PARK",
    "FLAGSHIP PARK",
    "CITY-STATE PARK",
    "STATE PARK",
    "STATE PARK PRESERVE",
    "NATURE AREA",
    "GARDEN",
    "NATURAL RESOURCE AREA",
    "HISTORIC HOUSE PARK",
    "BOTANICAL",
)

# Tag mapping from factype → user-friendly tag
_TYPE_TAGS: dict[str, str] = {
    "NEIGHBORHOOD PARK": "park",
    "COMMUNITY PARK": "park",
    "FLAGSHIP PARK": "park",
    "CITY-STATE PARK": "park",
    "STATE PARK": "park",
    "STATE PARK PRESERVE": "preserve",
    "NATURE AREA": "nature",
    "GARDEN": "garden",
    "NATURAL RESOURCE AREA": "nature",
    "HISTORIC HOUSE PARK": "historic",
    "BOTANICAL": "botanical",
}

# NYC bounding box for coord sanity-check
_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)


class ParksSource(DataSource):
    raw_count: int = 0
    bad_coords: int = 0

    @property
    def name(self) -> str:
        return "parks"

    @property
    def description(self) -> str:
        return "NYC Parks, Gardens & Green Spaces"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5"

    def fetch(self) -> list[Venue]:
        token = os.environ.get("NYC_OPEN_DATA_TOKEN", "")
        type_list = ",".join(f"'{t}'" for t in _PARK_TYPES)
        params: dict[str, Any] = {
            "$limit": 5000,
            "$where": (
                f"factype in({type_list}) "
                "AND latitude IS NOT NULL "
                "AND longitude IS NOT NULL"
            ),
        }
        if token:
            params["$$app_token"] = token

        log.info("Parks: fetching from Facilities Database …")
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        log.info("Parks: %d rows from API", len(rows))

        venues: list[Venue] = []
        for row in rows:
            self.raw_count += 1
            v = self._venue_from_row(row)
            if v:
                venues.append(v)
            else:
                self.bad_coords += 1

        log.info(
            "Parks: %d venues (from %d rows, %d bad coords)",
            len(venues), self.raw_count, self.bad_coords,
        )
        return venues

    @staticmethod
    def _venue_from_row(row: dict[str, Any]) -> Venue | None:
        fac_name = (row.get("facname") or "").strip()
        if not fac_name:
            return None

        try:
            lat = float(row["latitude"])
            lng = float(row["longitude"])
        except (KeyError, ValueError, TypeError):
            return None

        lat_lo, lat_hi, lng_lo, lng_hi = _NYC_BBOX
        if not (lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi):
            return None

        borough = (row.get("boro") or "").strip().upper()
        address = (row.get("address") or "").strip()
        zipcode = (row.get("zipcode") or "").strip()
        factype = (row.get("factype") or "").strip()

        tags: list[str] = []
        tag = _TYPE_TAGS.get(factype, "")
        if tag:
            tags.append(tag)

        meta: dict[str, Any] = {}
        opname = (row.get("opname") or "").strip()
        if opname:
            meta["operator"] = opname

        return Venue(
            name=fac_name.title(),
            lat=lat,
            lng=lng,
            source="parks",
            address=address,
            borough=borough,
            zipcode=zipcode,
            cuisine=factype.title(),
            tags=tags,
            meta=meta,
        )

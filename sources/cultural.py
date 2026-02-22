"""
NYC Museums, Galleries & Cultural Venues — from the City Planning Facilities DB.

Source: https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5
Includes museums, galleries, public art sites, historical societies,
architecture/design orgs, and multi-discipline cultural spaces funded
or registered through DCLA and other city agencies.

The DCLA Percent for Art public artworks dataset can be added as a
supplemental source once its Socrata 4×4 ID is confirmed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

# City Planning Facilities Database
FAC_DATASET_ID = "ji82-xba5"
FAC_BASE_URL = f"https://data.cityofnewyork.us/resource/{FAC_DATASET_ID}.json"

# Cultural facility types from the Facilities DB
_ART_TYPES = (
    "MUSEUM",
    "PUBLIC MUSEUMS AND SITES",
    "PRIVATE MUSEUMS AND SITES",
    "ARCHITECTURE/DESIGN",
    "CRAFTS",
    "PHOTOGRAPHY",
    "MULTI-DISCIPLINE, NON-PERFORM",
    "MULTI-DISCIPL, PERF & NON-PERF",
    "HISTORICAL SOCIETIES",
    "FOLK ARTS",
    "SCIENCE",
)

# Tag mapping
_TYPE_TAGS: dict[str, str] = {
    "MUSEUM": "museum",
    "PUBLIC MUSEUMS AND SITES": "museum",
    "PRIVATE MUSEUMS AND SITES": "museum",
    "ARCHITECTURE/DESIGN": "architecture",
    "CRAFTS": "crafts",
    "PHOTOGRAPHY": "photography",
    "MULTI-DISCIPLINE, NON-PERFORM": "gallery",
    "MULTI-DISCIPL, PERF & NON-PERF": "gallery",
    "HISTORICAL SOCIETIES": "historical",
    "FOLK ARTS": "folk_art",
    "SCIENCE": "science",
}

# NYC bounding box
_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)


class CulturalSource(DataSource):
    raw_count: int = 0
    bad_coords: int = 0

    @property
    def name(self) -> str:
        return "cultural"

    @property
    def description(self) -> str:
        return "NYC Museums, Galleries & Cultural Venues"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5"

    def fetch(self) -> list[Venue]:
        token = os.environ.get("NYC_OPEN_DATA_TOKEN", "")
        venues: list[Venue] = []

        # Facilities Database — cultural venues
        venues.extend(self._fetch_facilities(token))

        log.info("Cultural: %d total venues", len(venues))
        return venues

    def _fetch_facilities(self, token: str) -> list[Venue]:
        type_list = ",".join(f"'{t}'" for t in _ART_TYPES)
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

        log.info("Cultural: fetching Facilities Database …")
        resp = requests.get(FAC_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        log.info("Cultural: %d rows from Facilities DB", len(rows))

        venues: list[Venue] = []
        for row in rows:
            self.raw_count += 1
            v = self._venue_from_fac_row(row)
            if v:
                venues.append(v)
            else:
                self.bad_coords += 1

        log.info(
            "Cultural: %d venues from Facilities DB (from %d rows, %d bad coords)",
            len(venues), self.raw_count, self.bad_coords,
        )
        return venues

    @staticmethod
    def _venue_from_fac_row(row: dict[str, Any]) -> Venue | None:
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
        tag = _TYPE_TAGS.get(factype, "gallery")
        tags.append(tag)

        meta: dict[str, Any] = {}
        opname = (row.get("opname") or "").strip()
        if opname:
            meta["operator"] = opname

        return Venue(
            name=fac_name.title(),
            lat=lat,
            lng=lng,
            source="cultural",
            address=address,
            borough=borough,
            zipcode=zipcode,
            cuisine=factype.title(),
            tags=tags,
            meta=meta,
        )

"""
NYC Theater, Dance & Performing Arts — from the City Planning Facilities DB.

Source: https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5
DCLA-registered theater companies, multi-discipline performing arts
organizations, and dance companies across all five boroughs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

FAC_DATASET_ID = "ji82-xba5"
FAC_BASE_URL = f"https://data.cityofnewyork.us/resource/{FAC_DATASET_ID}.json"

_THEATER_TYPES = (
    "THEATER",
    "MULTI-DISCIPLINE, PERFORMING",
    "DANCE",
)

_TYPE_TAGS: dict[str, str] = {
    "THEATER": "theater",
    "MULTI-DISCIPLINE, PERFORMING": "performing_arts",
    "DANCE": "dance",
}

_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)


class TheaterSource(DataSource):
    @property
    def name(self) -> str:
        return "theaters"

    @property
    def description(self) -> str:
        return "NYC Theater, Dance & Performing Arts"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5"

    def fetch(self) -> list[Venue]:
        token = os.environ.get("NYC_OPEN_DATA_TOKEN", "")
        type_list = ",".join(f"'{t}'" for t in _THEATER_TYPES)
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

        log.info("Theater: fetching Facilities Database …")
        resp = requests.get(FAC_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        log.info("Theater: %d rows from Facilities DB", len(rows))

        venues: list[Venue] = []
        bad = 0
        for row in rows:
            v = self._venue_from_row(row)
            if v:
                venues.append(v)
            else:
                bad += 1
        log.info("Theater: %d venues (from %d rows, %d bad coords)", len(venues), len(rows), bad)
        return venues

    @staticmethod
    def _venue_from_row(row: dict[str, Any]) -> Venue | None:
        name = (row.get("facname") or "").strip()
        if not name:
            return None
        try:
            lat = float(row["latitude"])
            lng = float(row["longitude"])
        except (KeyError, ValueError, TypeError):
            return None

        lat_lo, lat_hi, lng_lo, lng_hi = _NYC_BBOX
        if not (lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi):
            return None

        factype = (row.get("factype") or "").strip()
        tags = [_TYPE_TAGS.get(factype, "theater")]

        meta: dict[str, Any] = {}
        opname = (row.get("opname") or "").strip()
        if opname:
            meta["operator"] = opname

        return Venue(
            name=name.title(),
            lat=lat,
            lng=lng,
            source="theaters",
            address=(row.get("address") or "").strip(),
            borough=(row.get("boro") or "").strip().upper(),
            zipcode=(row.get("zipcode") or "").strip(),
            cuisine=factype.title(),
            tags=tags,
            meta=meta,
        )

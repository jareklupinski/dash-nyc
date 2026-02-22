"""
NYC Playgrounds — from the City Planning Facilities Database.

Source: https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5
Includes PLAYGROUND and JOINTLY OPERATED PLAYGROUND facility types from
the DPR parks-properties data.  ~510 playgrounds across all five boroughs.
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

_PLAYGROUND_TYPES = ("PLAYGROUND", "JOINTLY OPERATED PLAYGROUND")

# NYC bounding box for coord sanity-check
_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)


class PlaygroundSource(DataSource):
    raw_count: int = 0
    bad_coords: int = 0

    @property
    def name(self) -> str:
        return "playgrounds"

    @property
    def description(self) -> str:
        return "NYC Parks Playgrounds"

    @property
    def url(self) -> str:
        return "https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5"

    def fetch(self) -> list[Venue]:
        token = os.environ.get("NYC_OPEN_DATA_TOKEN", "")
        type_list = ",".join(f"'{t}'" for t in _PLAYGROUND_TYPES)
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

        log.info("Playgrounds: fetching from Facilities Database …")
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        log.info("Playgrounds: %d rows from API", len(rows))

        venues: list[Venue] = []
        for row in rows:
            self.raw_count += 1
            v = self._venue_from_row(row)
            if v:
                venues.append(v)
            else:
                self.bad_coords += 1

        log.info(
            "Playgrounds: %d venues (from %d rows, %d bad coords)",
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
        if factype == "JOINTLY OPERATED PLAYGROUND":
            tags.append("jointly_operated")
        if factype == "PLAYGROUND":
            tags.append("playground")

        meta: dict[str, Any] = {}
        opname = (row.get("opname") or "").strip()
        if opname:
            meta["operator"] = opname

        return Venue(
            name=fac_name.title(),
            lat=lat,
            lng=lng,
            source="playgrounds",
            address=address,
            borough=borough,
            zipcode=zipcode,
            tags=tags,
            meta=meta,
        )

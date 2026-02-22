"""
SLA Bars — Filtered subset of NYS Liquor Authority active licenses.

Only includes venues tagged as bars (bar, tavern, club, cabaret).
Used by drink.dash.nyc to show the bar scene.

Source: https://lamp.sla.ny.gov/
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

# ArcGIS FeatureServer backing the LAMP web app
BASE_URL = (
    "https://services8.arcgis.com/kHNnQD79LvY0XnKy/arcgis/rest/services"
    "/ActiveLicensesV3/FeatureServer/0/query"
)
PAGE_SIZE = 16_000

COUNTY_TO_BORO = {
    "New York": "MANHATTAN",
    "Kings": "BROOKLYN",
    "Queens": "QUEENS",
    "Bronx": "BRONX",
    "Richmond": "STATEN ISLAND",
}

OUT_FIELDS = (
    "PremiseName,PremiseDBA,Description,PremiseAddress1,PremiseCity,"
    "CountyName,LicenseCla,LicensePermitID,PremiseZIP,Latitude,Longitude,"
    "Lic_Original_Date"
)

# License descriptions that indicate a bar / drinking establishment
_BAR_KEYWORDS = ("bar", "tavern", "club", "cabaret", "taproom", "lounge")


class SLABarsSource(DataSource):
    raw_count: int = 0
    dedup_count: int = 0

    @property
    def name(self) -> str:
        return "sla"

    @property
    def description(self) -> str:
        return "NYS Liquor Authority — Bars & Nightlife (NYC)"

    @property
    def url(self) -> str:
        return "https://lamp.sla.ny.gov/"

    def fetch(self) -> list[Venue]:
        """Fetch active bar/club liquor licenses in NYC from LAMP ArcGIS."""
        county_list = ",".join(f"'{c}'" for c in COUNTY_TO_BORO)
        where = (
            f"CountyName IN ({county_list}) "
            "AND Latitude IS NOT NULL AND Longitude IS NOT NULL"
        )

        venues: list[Venue] = []
        offset = 0

        while True:
            log.info("SLA Bars: fetching offset %d …", offset)

            params: dict[str, Any] = {
                "f": "json",
                "where": where,
                "outFields": OUT_FIELDS,
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
            }

            resp = requests.get(BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if not features:
                break

            for feat in features:
                attrs = feat.get("attributes", {})

                license_desc = attrs.get("Description", "")
                desc_lower = license_desc.lower()

                # Only keep bars / drinking establishments
                if not any(w in desc_lower for w in _BAR_KEYWORDS):
                    continue

                try:
                    lat = float(attrs.get("Latitude", 0))
                    lng = float(attrs.get("Longitude", 0))
                except (TypeError, ValueError):
                    continue

                if lat == 0 or lng == 0:
                    continue

                county = attrs.get("CountyName", "")
                borough = COUNTY_TO_BORO.get(county, county.upper())

                tags = ["liquor_license", "bar"]

                display_name = attrs.get("PremiseDBA") or attrs.get("PremiseName") or "Unknown"

                opened = ""
                orig_ts = attrs.get("Lic_Original_Date")
                if orig_ts:
                    try:
                        from datetime import datetime, timezone
                        opened = datetime.fromtimestamp(orig_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                    except (ValueError, TypeError, OSError):
                        pass

                venues.append(
                    Venue(
                        name=display_name.title(),
                        lat=lat,
                        lng=lng,
                        source=self.name,
                        address=attrs.get("PremiseAddress1", ""),
                        borough=borough,
                        zipcode=attrs.get("PremiseZIP", ""),
                        tags=tags,
                        meta={
                            "license_type": license_desc,
                            "license_id": attrs.get("LicensePermitID", ""),
                        },
                        opened=opened,
                    )
                )

            offset += len(features)
            if not data.get("exceededTransferLimit", False):
                break

        self.raw_count = len(venues)

        # Dedup: same establishment can hold multiple license types
        deduped: dict[tuple[str, str, str], Venue] = {}
        for v in venues:
            key = (v.name.lower().strip(), v.address.lower().strip(), v.borough)
            if key in deduped:
                existing = deduped[key]
                for t in v.tags:
                    if t not in existing.tags:
                        existing.tags.append(t)
                existing.meta.setdefault("all_license_types", [existing.meta.get("license_type", "")])
                if v.meta.get("license_type") not in existing.meta["all_license_types"]:
                    existing.meta["all_license_types"].append(v.meta.get("license_type", ""))
            else:
                deduped[key] = v

        venues = list(deduped.values())
        self.dedup_count = self.raw_count - len(venues)
        log.info("SLA Bars: fetched %d bar venues (%d duplicates removed)", len(venues), self.dedup_count)
        return venues

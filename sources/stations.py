"""
NYC Weather Stations — observations from the National Weather Service API.

Source: https://api.weather.gov
Fetches current weather conditions from NWS observation stations within NYC,
including temperature, wind, humidity, and conditions text.
"""

from __future__ import annotations

import logging
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from lib.source_base import DataSource, Venue

log = logging.getLogger(__name__)

_NWS_HEADERS = {
    "User-Agent": "(dash.nyc, mail@dash.nyc)",
    "Accept": "application/geo+json",
}

# NYC bounding box
_NYC_BBOX = (40.49, 40.92, -74.27, -73.68)


class WeatherStationsSource(DataSource):
    @property
    def name(self) -> str:
        return "weather_stations"

    @property
    def description(self) -> str:
        return "NWS Weather Observation Stations in NYC"

    @property
    def url(self) -> str:
        return "https://www.weather.gov"

    def fetch(self) -> list[Venue]:
        stations = self._discover_stations()
        log.info("Weather: %d stations in NYC bbox", len(stations))

        observations = self._fetch_observations(stations)
        log.info("Weather: %d observations fetched", len(observations))

        venues: list[Venue] = []
        for sid, stn in stations.items():
            obs = observations.get(sid)
            v = self._make_venue(stn, obs)
            if v:
                venues.append(v)

        log.info("Weather: %d venues total", len(venues))
        return venues

    def _discover_stations(self) -> dict[str, dict]:
        """Fetch observation stations near NYC via OKX grid point."""
        # OKX/33,35 is the NYC (Manhattan) grid point
        url = "https://api.weather.gov/gridpoints/OKX/33,35/stations"
        stations: dict[str, dict] = {}
        lat_lo, lat_hi, lng_lo, lng_hi = _NYC_BBOX

        try:
            resp = requests.get(url, headers=_NWS_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("Weather: station discovery failed: %s", exc)
            return stations

        for feat in data.get("features", []):
            coords = feat.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue
            lng, lat = coords[0], coords[1]
            if not (lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi):
                continue
            props = feat.get("properties", {})
            sid = props.get("stationIdentifier", "")
            if sid:
                stations[sid] = {
                    "id": sid,
                    "name": props.get("name", sid),
                    "lat": lat,
                    "lng": lng,
                    "elevation_m": (props.get("elevation") or {}).get("value"),
                }

        return stations

    def _fetch_observations(self, stations: dict[str, dict]) -> dict[str, dict]:
        """Fetch latest observation for each station in parallel."""
        obs: dict[str, dict] = {}

        def _get_obs(sid: str) -> tuple[str, dict | None]:
            try:
                url = f"https://api.weather.gov/stations/{sid}/observations/latest"
                resp = requests.get(url, headers=_NWS_HEADERS, timeout=10)
                if resp.status_code != 200:
                    return sid, None
                data = resp.json()
                return sid, data.get("properties", {})
            except Exception:
                return sid, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_get_obs, sid): sid for sid in stations}
            for future in as_completed(futures):
                sid, result = future.result()
                if result:
                    obs[sid] = result

        return obs

    @staticmethod
    def _make_venue(station: dict, obs: dict | None) -> Venue | None:
        lat = station.get("lat")
        lng = station.get("lng")
        if lat is None or lng is None:
            return None

        name = station.get("name", station.get("id", "Unknown"))
        sid = station.get("id", "")

        tags: list[str] = ["weather_station"]
        meta: dict[str, Any] = {"station_id": sid}

        # Extract observation data
        if obs:
            temp_c = _safe_val(obs, "temperature")
            if temp_c is not None:
                temp_f = round(temp_c * 9 / 5 + 32, 1)
                meta["temp_f"] = temp_f
                meta["temp_c"] = round(temp_c, 1)
                tags.append(f"{temp_f}°F")

            desc = (obs.get("textDescription") or "").strip()
            if desc:
                meta["conditions"] = desc
                tags.append(desc.lower())

            wind_ms = _safe_val(obs, "windSpeed")
            if wind_ms is not None:
                wind_mph = round(wind_ms * 2.237, 1)
                meta["wind_mph"] = wind_mph

            humidity = _safe_val(obs, "relativeHumidity")
            if humidity is not None:
                meta["humidity"] = round(humidity, 1)

        cuisine = meta.get("conditions", "Weather Station")

        return Venue(
            name=name,
            lat=lat,
            lng=lng,
            source="weather_stations",
            address=f"Station {sid}",
            tags=tags,
            cuisine=cuisine,
            meta=meta,
        )


def _safe_val(obs: dict, key: str) -> float | None:
    """Extract a numeric value from NWS observation nested structure."""
    entry = obs.get(key)
    if isinstance(entry, dict):
        v = entry.get("value")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None

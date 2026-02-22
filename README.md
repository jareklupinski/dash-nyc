DANGER! MOVING PARTS! Keep fingers clear at all times!

<div align="center">
  <picture>
    <img src="./shared/static/warning.svg" alt="Warning: Moving Parts" width="160" />
  </picture>
</div>

This repository is only meant to be consumed and manipulated by AI agents.

Humans proceeding with their own eyes and fingers do so at their own risk!

# dash-nyc

An interactive map dashboard for New York City built from open data.
Every category (eat, drink, art, …) is an app served at **dash.nyc/\<app\>/**,
with subdomains like `eat.dash.nyc` 301-redirecting there.

**Live:** [dash.nyc](https://dash.nyc)

## Architecture

```
┌───────────────┐   rsync    ┌───────────────┐
│  build server │ ──────────►│  web server    │
│               │  dist/*    │                │
│               │            │                │
│  • cron timer │            │  • static only │
│  • build.py   │            │  • SSL/certbot │
│  • deploy.py  │            │  • nginx       │
│  • crossref   │            │                │
└───────────────┘            └───────────────┘
```

### Portal (SPA)

The **dash** app is a single-page application that serves all categories.
When a user visits `dash.nyc/eat/`, nginx falls through to the dash SPA
template (`apps/dash/templates/index.html`), which fetches
`/eat/venues.json` + `/eat/venue_info.json` dynamically and builds
the filter UI from server-provided data.

Subdomain redirects (`eat.dash.nyc → dash.nyc/eat/`) are configured in
nginx.  Each app also has a standalone template in `apps/<id>/templates/`
used by the standalone subdomain builds, but the canonical entry point is
always the dash SPA.

### Staging / production

```
~/dash-nyc/
  releases/<sha>/            ← immutable, SHA-named release dirs
      dash.nyc/              ← portal + per-category data (eat/, drink/, …)
      eat.dash.nyc/          ← standalone subdomain copies
      …
  staging    → releases/<sha>   (served by staging.dash.nyc)
  production → releases/<sha>   (served by dash.nyc)
```

`deploy.py` rsyncs `dist/*` into a new release dir, flips the `staging`
symlink, runs smoke tests against `staging.dash.nyc`, then (with
`--promote-automatically`) flips `production`.

### Deterministic build SHA

There is no `.git` directory.  `build.py` computes a deterministic 8-char
SHA by hashing all source files (`.py`, `.html`, `.yaml`, `.css`, `.js`,
`.json`, `.in`, `.md`, `Makefile`) using git-blob-format SHA-1.  The SHA
changes only when source files change.

### Unified filter system

`build.py` is the **single source of truth** for all filter logic.
`_compute_venue_filters()` pre-computes a `_ft` (filter tags) array on
every venue at build time and emits `filters` + `filter_labels` in both
`venues.js` (for standalone templates) and `venues.json` (for the SPA).
The dash template reads these directly — it never computes filter
membership client-side.

Per-app filter rules (defined in `_APP_EXTRA_FILTERS`):

| App | Filters |
|-----|---------|
| eat | diet tags, Google Gems, Yelp Gems, New, Not on Yelp, Not on Google, Park Food |
| drink | Bars, Coffee & Tea, Juice & Frozen, New |
| food | Farmers Market, Grocery, Grocery (EBT) |
| news | Good News, Bad News, + topic tags |
| others | raw source tags from the data |

## Apps

17 apps, each defined by `apps/<id>/app.yaml`:

| App | Domain | Sources | Venues | Features |
|-----|--------|---------|--------|----------|
| **dash** | dash.nyc | news | ~100 | SPA portal, landing shows good-news markers |
| **eat** | eat.dash.nyc | dohmh, sla, dpr | ~40k | Cross-ref (Yelp/Google/OpenTable), dietary tags, cross-source merge |
| **drink** | drink.dash.nyc | sla_bars, coffee | ~3k | Cross-ref (Yelp/Google), type filters |
| **food** | food.dash.nyc | grocery, greenmarket | ~11k | Source-type filters (EBT/SNAP) |
| **news** | *(portal only)* | news | ~100 | Sentiment filter, topic tags |
| **art** | art.dash.nyc | cultural | ~600 | Museums, galleries |
| **shop** | shop.dash.nyc | shops | ~6.5k | Retail & pop-ups |
| **theater** | theater.dash.nyc | theaters | ~700 | Dance, performing arts |
| **play** | play.dash.nyc | playgrounds | ~500 | |
| **movies** | movies.dash.nyc | cinemas | ~100 | |
| **music** | music.dash.nyc | musicians | ~280 | |
| **make** | make.dash.nyc | workshops | ~350 | Libraries, makerspaces |
| **relax** | relax.dash.nyc | parks | ~900 | Gardens, nature preserves |
| **pee** | pee.dash.nyc | restrooms | ~420 | Accessibility, seasonal filters |
| **weather** | weather.dash.nyc | stations | 5 | NWS observation stations |
| **today** | today.dash.nyc | news, greenmarket | ~270 | Time-sensitive aggregator (1h refresh) |
| **vibe** | vibe.dash.nyc | *(all 18 sources)* | ~97k | Heatmap-only, category weight sliders |

## Project structure

```
build.py            — builds all 17 apps, outputs dist/<id>/
deploy.py           — rsync + staging/production symlinks + smoke tests
Makefile            — orchestrates build, deploy, nginx, certbot, timers
nginx.conf.in       — nginx template (all domains in one server block)
requirements.txt    — Python deps: requests, jinja2, pyyaml, beautifulsoup4

sources/            — 18 data-source modules (one per API)
  dohmh.py          — DOHMH restaurant inspections (~27k)
  sla.py            — SLA liquor licenses (~22k, for eat merge)
  sla_bars.py       — SLA bars subset (~1.2k, for drink)
  coffee.py         — DOHMH coffee/tea/juice shops (~1.7k)
  grocery.py        — NYS grocery stores + SNAP/EBT (~8k)
  greenmarket.py    — GrowNYC farmers markets (~140)
  dpr.py            — Parks Dept food vendors (~60)
  news.py           — NYC news RSS feeds, geocoded (~100)
  cultural.py       — Museums, galleries (~600)
  shops.py          — Licensed retail (~6.5k)
  theaters.py       — Theater & performing arts (~700)
  playgrounds.py    — Parks playgrounds (~500)
  cinemas.py        — Film orgs (~100)
  musicians.py      — Music orgs (~280)
  workshops.py      — Libraries & makerspaces (~350)
  parks.py          — Parks & green spaces (~900)
  restrooms.py      — Public restrooms (~420)
  stations.py       — NWS weather stations (5)

lib/
  source_base.py    — Venue dataclass + DataSource ABC (auto-discovered)
  crossref.py       — Yelp/Google Places cross-referencing (SQLite cache)
  diet_sources.py   — HMS Halal + KosherNearMe dietary-tag fetching

apps/
  <id>/
    app.yaml        — app config: title, domains, sources, colors, crossref, diet, merge
    templates/
      index.html    — standalone Jinja2 template
    static/         — favicon, OG images (optional)
  dash/
    templates/
      index.html    — SPA portal template (Leaflet + Supercluster inlined)

shared/
  static/
    style.css       — global TUI-styled CSS
    dash-transition.js
    favicon.svg
    og-image.png

data/
  crossref.db       — persistent SQLite cross-ref cache (not in repo)

cron/
  dash-nyc-refresh       — daily rebuild script
  dash-nyc-refresh.service.in — systemd service template
  dash-nyc-refresh.timer — runs at 08:00 UTC (3:00 AM ET)

vendor/
  TuiCss/           — TUI CSS framework (MIT)
```

## Build output

Per app, `build.py` writes to `dist/<id>/`:

| File | Purpose |
|------|---------|
| `venues.js` | `VENUE_DATA`, `SOURCE_META`, `ALL_TAGS`, `ALL_SOURCES`, `ALL_DIETS`, `DIET_SOURCE_STATS`, `FILTERS`, `FILTER_LABELS_MAP` |
| `venues.json` | Same data as JSON (for SPA `fetch()`) — includes `filters` + `filter_labels` keys |
| `venue_info.js` | `VENUE_INFO` — pre-rendered HTML panels (index-matched to VENUE_DATA) |
| `venue_info.json` | Same as JSON (for SPA `fetch()`) |
| `heatpoints.js` | Vibe app only — compact lat/lng/category arrays |
| `index.html` | Rendered Jinja2 template |
| `manifest.json` | Build manifest: `sha`, `built_at`, `venue_count`, `data_hash` |
| `source_info.json` | Lightweight source list for portal aggregation |
| `style.css` | Copied from `shared/static/` |

Each venue in the data carries a pre-computed `_ft` array (filter tags).
The `filters` list defines the ordered set of filter buttons shown in the UI.
The `filter_labels` map provides display names.

## Development

```bash
# Install dependencies
make install

# Build one app (fresh fetch)
make build APP=eat

# Build one app (cached data, skip API calls)
make build-cached APP=eat

# Build all 17 apps
make build-all-cached

# Local dev server
make serve APP=eat    # → http://localhost:8000

# Deploy one app
make deploy APP=eat

# Deploy all apps
make deploy-all

# Staging only (no auto-promote)
make stage

# Promote staging → production
make promote

# Rollback production to previous release
make rollback

# Check current staging/production SHAs
make status
```

### Adding a new app

1. Create a source module in `sources/<name>.py` subclassing `DataSource`
2. Create `apps/<id>/` with:
   - `app.yaml` — config (id, title, domains, sources list, colors, etc.)
   - `templates/index.html` — standalone Jinja2 template
   - `static/` — favicon, OG images (optional)
3. Add the app to the `sub_apps` list in `apps/dash/app.yaml` so it
   appears in the portal nav
4. Add the subdomain server block to `nginx.conf.in` (301 → `dash.nyc/<id>/`)
5. Build and deploy:
   ```bash
   make build APP=<id>
   make deploy-all
   make deploy-nginx
   make certbot        # add new domain to cert
   ```

## Deployment

### .env

```bash
WEB_HOST=user@web.server
BUILD_HOST=user@build.server
BUILD_REPO=/home/user/dash-nyc
YELP_API_KEY=<key>
```

### Make targets

| Target | Where | Description |
|--------|-------|-------------|
| `build` | local | Build one app (`APP=eat`) |
| `build-all-cached` | local | Build all 17 apps from cache |
| `deploy` | → web | Build + rsync + smoke test + promote |
| `deploy-all` | → web | Same for all apps |
| `deploy-only` | → web | Rsync without rebuild |
| `stage` | → web | Deploy to staging only |
| `promote` | → web | Promote staging → production |
| `rollback` | → web | Swap production to previous release |
| `status` | → web | Show staging/production SHAs |
| `sync-repo` | → build | Rsync project code to build server |
| `deploy-nginx` | → web | Generate + install nginx config |
| `certbot` | → web | Issue/renew multi-domain cert (HTTP-01) |
| `certbot-wildcard` | → web | Issue wildcard cert (DNS-01, Cloudflare) |
| `timer-install` | → build | Install systemd timer |
| `timer-remove` | → build | Remove systemd timer |

### Daily refresh (systemd timer)

Fires at 08:00 UTC (3:00 AM ET) on the build server:

1. Full build of all apps (fresh API fetch)
2. If `YELP_API_KEY` set: runs cross-reference, then rebuilds with `--cache`
3. Deploys all apps via rsync to web server

### Cross-referencing

`lib/crossref.py` enriches eat/drink venues against Yelp & Google Places
APIs.  Results are stored in `data/crossref.db` (SQLite) and survive
across builds.  `stamp_crossref()` in `build.py` reads the DB at build
time and adds per-venue fields: `xr_y`, `xr_g` (found/not_found),
`yr`/`yrt` (Yelp reviews/rating), `gr`/`grt` (Google reviews/rating),
plus OpenTable links.

### Dietary tagging

`lib/diet_sources.py` fetches authoritative halal (HMS USA) and kosher
(KosherNearMe) certification lists.  `apply_diet_tags()` in `build.py`
matches venues by name/address/geo, then supplements with DOHMH cuisine
fields and Yelp category data.  Currently enabled for the **eat** app only.

## SSL Certificates

All domains served via Let's Encrypt, managed by certbot on the web server.

```bash
# Multi-domain cert (HTTP-01, any DNS provider)
make certbot

# Wildcard cert (DNS-01, requires Cloudflare DNS)
make certbot-wildcard
```

## License

Data from NYC Open Data is public domain.
TUI CSS framework: MIT (see `vendor/TuiCss/LICENSE.md`).
All else Unlicensed.

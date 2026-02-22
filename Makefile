.PHONY: build build-cached build-all serve deploy deploy-only deploy-all \
       sync-repo deploy-nginx timer-install timer-remove \
       clean clean-all promote rollback status stage

# Load deployment settings from .env
-include .env

PYTHON ?= python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

# Two-server model:
#   BUILD_HOST = where builds + cron run
#   WEB_HOST   = static file server only
BUILD_HOST ?= user@build-host
BUILD_REPO ?= /home/user/dash-nyc
WEB_HOST   ?= user@web-host

# Default app to build (override: make build APP=art)
APP ?= food

# --- Setup ---

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $@

install: $(VENV)/bin/activate

# --- Build ---

build: install
	$(PY) build.py $(APP) -v

build-cached: install
	$(PY) build.py $(APP) --cache -v

build-all: install
	$(PY) build.py --all -v

build-all-cached: install
	$(PY) build.py --all --cache -v

# --- Dev ---

serve: build
	cd dist/$(APP) && $(PYTHON) -m http.server 8000

# --- Deploy ---
# deploy.py rsyncs to staging, smoke tests staging.dash.nyc, then optionally promotes

deploy: build
	$(PY) deploy.py $(APP) -v --promote-automatically

deploy-only:
	$(PY) deploy.py $(APP) -v --promote-automatically

deploy-all: build-all
	$(PY) deploy.py --all -v --promote-automatically

# Deploy to staging only (no auto-promote)
stage: build-all
	$(PY) deploy.py --all -v

stage-only:
	$(PY) deploy.py --all -v

promote:
	$(PY) deploy.py --promote -v

rollback:
	$(PY) deploy.py --rollback -v

status:
	$(PY) deploy.py --status -v

smoke-test:
	$(PY) deploy.py $(APP) --smoke-only -v

# Sync the project repo to the BUILD server (for cron/timer refreshes)
sync-repo:
	rsync -avz --delete \
	  --exclude='.venv/' \
	  --exclude='dist/' \
	  --exclude='__pycache__/' \
	  --exclude='apps/*/\.cache/*.json' \
	  --exclude='apps/*/\.cache/*.log' \
	  --exclude='apps/*/\.cache/*.zip' \
	  --exclude='.env' \
	  --exclude='.git/' \
	  --exclude='*.pyc' \
	  ./ $(BUILD_HOST):$(BUILD_REPO)/
	@echo "Repo synced to build server."

# --- Nginx (runs on WEB_HOST) ---

WEB_USER := $(firstword $(subst @, ,$(WEB_HOST)))
WEB_PATH ?= /home/$(WEB_USER)/dash-nyc/production
STAGING_PATH ?= /home/$(WEB_USER)/dash-nyc/staging

NGINX_CONF   := dash-nyc.conf
NGINX_REMOTE  := /home/$(WEB_USER)/dash-nyc/$(NGINX_CONF)

CERT_PATH ?= /etc/letsencrypt/live/dash.nyc

nginx.conf: nginx.conf.in
	sed -e 's|{{PRODUCTION_PATH}}|$(WEB_PATH)|g' \
	    -e 's|{{STAGING_PATH}}|$(STAGING_PATH)|g' \
	    -e 's|{{CERT_PATH}}|$(CERT_PATH)|g' $< > $@
	@echo "Generated $@"

deploy-nginx: nginx.conf
	rsync -avz nginx.conf $(WEB_HOST):$(NGINX_REMOTE)
	ssh $(WEB_HOST) '\
	  sudo ln -sf $(NGINX_REMOTE) /etc/nginx/sites-enabled/$(NGINX_CONF) && \
	  sudo nginx -t && sudo systemctl reload nginx'
	@echo "Deployed nginx config (all dash.nyc domains)"

# Standalone domains not tied to any app (served by the same nginx)
EXTRA_DOMAINS := staging.dash.nyc

# Collect every domain from every app.yaml + extras into a single cert
APP_DOMAINS := $(shell $(PY) -c "\
import yaml, pathlib; \
ds = set(); \
[ds.update(yaml.safe_load(f.read_text()).get('app',{}).get('domains',[])) for f in pathlib.Path('apps').glob('*/app.yaml')]; \
print(' '.join(sorted(ds)))" 2>/dev/null)
ALL_DOMAINS := $(sort $(APP_DOMAINS) $(EXTRA_DOMAINS))

# Option A: Multi-domain cert via HTTP-01 (works with any DNS provider)
certbot:
	@echo "Issuing cert for: $(ALL_DOMAINS)"
	ssh $(WEB_HOST) 'sudo certbot --nginx $(foreach d,$(ALL_DOMAINS),-d $(d)) --expand'

# Option B: Wildcard cert via DNS-01 (requires Cloudflare DNS)
# 1. Point dash.nyc nameservers to Cloudflare (free plan)
# 2. Create API token: cloudflare.com → My Profile → API Tokens → Create Token
#    (Zone:DNS:Edit for the zone)
# 3. On WEB_HOST:
#      sudo apt install python3-certbot-dns-cloudflare
#      echo "dns_cloudflare_api_token = <token>" | sudo tee /etc/letsencrypt/cloudflare.ini
#      sudo chmod 600 /etc/letsencrypt/cloudflare.ini
# Then run:
certbot-wildcard:
	ssh $(WEB_HOST) 'sudo certbot certonly --dns-cloudflare \
	  --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini \
	  -d "dash.nyc" -d "*.dash.nyc" \
	  --preferred-challenges dns-01'
	@echo "Wildcard certs issued. Run: make deploy-nginx to update nginx."

# --- Timer (runs on BUILD_HOST) ---

cron/dash-nyc-refresh.service: cron/dash-nyc-refresh.service.in .env
	sed -e 's|{{BUILD_REPO}}|$(BUILD_REPO)|g' \
	    -e 's|{{BUILD_USER}}|$(firstword $(subst @, ,$(BUILD_HOST)))|g' \
	    -e 's|{{YELP_API_KEY}}|$(YELP_API_KEY)|g' $< > $@
	@echo "Generated $@"

timer-install: cron/dash-nyc-refresh.service
	rsync -avz cron/ $(BUILD_HOST):$(BUILD_REPO)/cron/
	ssh $(BUILD_HOST) 'chmod +x $(BUILD_REPO)/cron/dash-nyc-refresh && \
	  sudo cp $(BUILD_REPO)/cron/dash-nyc-refresh.service /etc/systemd/system/ && \
	  sudo cp $(BUILD_REPO)/cron/dash-nyc-refresh.timer /etc/systemd/system/ && \
	  sudo systemctl daemon-reload && \
	  sudo systemctl enable --now dash-nyc-refresh.timer'
	@echo "Timer installed on build server."

timer-remove:
	ssh $(BUILD_HOST) 'sudo systemctl disable --now dash-nyc-refresh.timer && \
	  sudo rm -f /etc/systemd/system/dash-nyc-refresh.{service,timer} && \
	  sudo systemctl daemon-reload'
	@echo "Timer removed from build server."

# --- Clean ---

clean:
	rm -rf dist

clean-all:
	rm -rf dist apps/*/.cache

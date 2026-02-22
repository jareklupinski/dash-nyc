#!/usr/bin/env python3
"""
deploy.py — Deploy built apps to the web server with staging/promotion.

Architecture:
    /home/<user>/dash-nyc/
        releases/<sha>/            ← SHA-named release directories
            dash.nyc/              ← main app (with eat/, drink/, food/ sub-apps)
            drink.dash.nyc/        ← subdomain redirect targets
            eat.dash.nyc/
            food.dash.nyc/
        staging    → releases/<sha>  ← symlink, served by staging.dash.nyc
        production → releases/<sha>  ← symlink, served by dash.nyc

Usage:
    python deploy.py --all -v                          # deploy to staging, smoke test, wait
    python deploy.py --all -v --promote-automatically  # deploy, smoke, auto-promote if green
    python deploy.py --promote -v                      # promote current staging to production
    python deploy.py --status -v                       # show current staging/production SHAs
    python deploy.py --rollback -v                     # swap production back to previous release
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import requests

log = logging.getLogger("deploy")

ROOT = Path(__file__).parent
DIST = ROOT / "dist"

# Remote base directory for all releases
REMOTE_BASE = "dash-nyc"


def load_deploy_env() -> dict[str, str]:
    """Load deployment settings from .env file."""
    env = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess:
    """Run a command on the remote host via SSH."""
    return subprocess.run(
        ["ssh", host, cmd],
        capture_output=True, text=True,
    )


def get_build_sha() -> str:
    """Get the 8-char SHA for the current build from dist/dash/manifest.json or git."""
    manifest = DIST / "dash" / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        sha = data.get("sha", "")
        if sha and sha != "unknown":
            return sha[:8]
    # Fall back to git
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    # Last resort: hash the dist directory
    import hashlib, time
    return hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:8]


def get_remote_link(host: str, link_name: str) -> str | None:
    """Read a symlink target on the remote server. Returns the SHA or None."""
    user = host.split("@")[0]
    result = _ssh(host, f"readlink /home/{user}/{REMOTE_BASE}/{link_name} 2>/dev/null")
    if result.returncode != 0 or not result.stdout.strip():
        return None
    target = result.stdout.strip()
    # target is like "releases/a1b2c3d4" — extract the SHA
    if "/" in target:
        return target.split("/")[-1]
    return target


def deploy_release(sha: str, dry_run: bool = False) -> bool:
    """Rsync all built apps into a release directory, then point staging at it."""
    env = load_deploy_env()
    vps_host = env.get("WEB_HOST", "user@host")
    user = vps_host.split("@")[0]
    base = f"/home/{user}/{REMOTE_BASE}"
    release_dir = f"{base}/releases/{sha}"

    # Iterate all built apps and rsync each to the release dir
    app_ids = sorted(
        d.name for d in DIST.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )
    if not app_ids:
        log.error("No built apps found in dist/")
        return False

    # Create release directory
    result = _ssh(vps_host, f"mkdir -p {release_dir}")
    if result.returncode != 0:
        log.error("Failed to create release dir: %s", result.stderr)
        return False

    for app_id in app_ids:
        dist_dir = DIST / app_id
        manifest = json.loads((dist_dir / "manifest.json").read_text())
        domains = manifest.get("domains", [])
        if not domains:
            log.warning("No domains for %s, skipping", app_id)
            continue

        for domain in domains:
            target = f"{release_dir}/{domain}"
            log.info("Deploying %s → %s:%s (sha=%s)", app_id, vps_host, target, sha)

            cmd = [
                "rsync", "-avz", "--delete",
                "--exclude=nginx.conf",
                "--exclude=refresh.log",
            ]
            if dry_run:
                cmd.append("--dry-run")
            cmd.extend([f"{dist_dir}/", f"{vps_host}:{target}/"])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.error("rsync failed for %s:\n%s", domain, result.stderr)
                return False
            log.info("  Synced %s", domain)
            if result.stdout:
                for line in result.stdout.strip().split("\n")[-3:]:
                    log.info("    %s", line)

    if dry_run:
        log.info("Dry run complete — not updating symlinks")
        return True

    # Point staging symlink at this release
    symlink = f"{base}/staging"
    result = _ssh(vps_host, f"ln -sfn releases/{sha} {symlink}")
    if result.returncode != 0:
        log.error("Failed to update staging symlink: %s", result.stderr)
        return False
    log.info("staging → releases/%s", sha)

    return True


def promote(host: str | None = None) -> bool:
    """Promote staging to production by flipping the production symlink."""
    env = load_deploy_env()
    vps_host = host or env.get("WEB_HOST", "user@host")
    user = vps_host.split("@")[0]
    base = f"/home/{user}/{REMOTE_BASE}"

    staging_sha = get_remote_link(vps_host, "staging")
    if not staging_sha:
        log.error("No staging release found — deploy first")
        return False

    prod_sha = get_remote_link(vps_host, "production")
    if prod_sha == staging_sha:
        log.info("Production is already at %s — nothing to do", staging_sha)
        return True

    # Save current production as previous (for rollback)
    if prod_sha:
        _ssh(vps_host, f"ln -sfn releases/{prod_sha} {base}/previous")

    # Point production at staging's release
    result = _ssh(vps_host, f"ln -sfn releases/{staging_sha} {base}/production")
    if result.returncode != 0:
        log.error("Failed to update production symlink: %s", result.stderr)
        return False

    # Update legacy domain symlinks to point at the production release
    release_dir = f"{base}/releases/{staging_sha}"
    result = _ssh(vps_host, f"ls -1 {release_dir} 2>/dev/null")
    if result.returncode == 0 and result.stdout.strip():
        for domain_dir in result.stdout.strip().split("\n"):
            legacy = f"/home/{user}/{domain_dir}"
            target = f"{release_dir}/{domain_dir}"
            _ssh(vps_host, f"rm -rf {legacy} && ln -sfn {target} {legacy}")
            log.info("  %s → %s", domain_dir, target)

    log.info("Promoted: production → releases/%s (was: %s)", staging_sha, prod_sha or "none")

    # Reload nginx to pick up any path changes
    _ssh(vps_host, "sudo nginx -t && sudo systemctl reload nginx")
    return True


def rollback() -> bool:
    """Swap production back to the previous release."""
    env = load_deploy_env()
    vps_host = env.get("WEB_HOST", "user@host")
    user = vps_host.split("@")[0]
    base = f"/home/{user}/{REMOTE_BASE}"

    prev_sha = get_remote_link(vps_host, "previous")
    if not prev_sha:
        log.error("No previous release to roll back to")
        return False

    prod_sha = get_remote_link(vps_host, "production")
    result = _ssh(vps_host, f"ln -sfn releases/{prev_sha} {base}/production")
    if result.returncode != 0:
        log.error("Rollback failed: %s", result.stderr)
        return False

    # Update legacy domain symlinks
    release_dir = f"{base}/releases/{prev_sha}"
    result = _ssh(vps_host, f"ls -1 {release_dir} 2>/dev/null")
    if result.returncode == 0 and result.stdout.strip():
        for domain_dir in result.stdout.strip().split("\n"):
            legacy = f"/home/{user}/{domain_dir}"
            target = f"{release_dir}/{domain_dir}"
            _ssh(vps_host, f"rm -rf {legacy} && ln -sfn {target} {legacy}")

    # Swap previous to old production
    if prod_sha:
        _ssh(vps_host, f"ln -sfn releases/{prod_sha} {base}/previous")

    log.info("Rolled back: production → releases/%s (was: %s)", prev_sha, prod_sha or "none")
    _ssh(vps_host, "sudo nginx -t && sudo systemctl reload nginx")
    return True


def show_status() -> None:
    """Show current staging and production SHAs."""
    env = load_deploy_env()
    vps_host = env.get("WEB_HOST", "user@host")
    user = vps_host.split("@")[0]
    base = f"/home/{user}/{REMOTE_BASE}"

    staging_sha = get_remote_link(vps_host, "staging")
    prod_sha = get_remote_link(vps_host, "production")
    prev_sha = get_remote_link(vps_host, "previous")

    log.info("staging:    %s", staging_sha or "(none)")
    log.info("production: %s", prod_sha or "(none)")
    log.info("previous:   %s", prev_sha or "(none)")

    # List recent releases
    result = _ssh(vps_host, f"ls -1t {base}/releases/ 2>/dev/null | head -10")
    if result.returncode == 0 and result.stdout.strip():
        log.info("Recent releases:")
        for r in result.stdout.strip().split("\n"):
            marker = ""
            if r == staging_sha:
                marker = " ← staging"
            if r == prod_sha:
                marker += " ← production"
            if r == prev_sha:
                marker += " ← previous"
            log.info("  %s%s", r, marker)


def cleanup_releases(keep: int = 5) -> None:
    """Remove old releases, keeping the N most recent plus any active ones."""
    env = load_deploy_env()
    vps_host = env.get("WEB_HOST", "user@host")
    user = vps_host.split("@")[0]
    base = f"/home/{user}/{REMOTE_BASE}"

    # Get active SHAs
    active = set()
    for link in ("staging", "production", "previous"):
        sha = get_remote_link(vps_host, link)
        if sha:
            active.add(sha)

    # List all releases by modification time (newest first)
    result = _ssh(vps_host, f"ls -1t {base}/releases/ 2>/dev/null")
    if result.returncode != 0 or not result.stdout.strip():
        return

    releases = result.stdout.strip().split("\n")
    to_keep = set(releases[:keep]) | active
    to_remove = [r for r in releases if r not in to_keep]

    for r in to_remove:
        log.info("Removing old release: %s", r)
        _ssh(vps_host, f"rm -rf {base}/releases/{r}")


# --------------------------------------------------------------------------
# Smoke tests
# --------------------------------------------------------------------------
def smoke_test_domain(domain: str, manifest: dict) -> tuple[int, int]:
    """Run smoke tests against a single domain. Returns (passed, failed)."""
    passed = 0
    failed = 0

    app_id = manifest.get("app_id", "")
    is_heatmap = app_id in ("vibe",)  # heatmap-only apps use heatpoints.js

    # Test 1: index.html loads
    try:
        r = requests.get(f"https://{domain}/index.html", timeout=15, allow_redirects=True)
        # heatmap apps reference heatpoints.js instead of venues.js
        expected_content = "heatpoints.js" if is_heatmap else ("VENUE_DATA" in r.text or "venues.js" in r.text)
        if is_heatmap:
            content_ok = r.status_code == 200 and "heatpoints.js" in r.text
        else:
            content_ok = r.status_code == 200 and ("VENUE_DATA" in r.text or "venues.js" in r.text)
        if content_ok:
            log.info("  [PASS] index.html loads (%d bytes)", len(r.content))
            passed += 1
        else:
            log.error("  [FAIL] index.html: status=%d, missing expected content", r.status_code)
            failed += 1
    except Exception as e:
        log.error("  [FAIL] index.html: %s", e)
        failed += 1

    # Test 2: data file loads (venues.js or heatpoints.js for heatmap apps)
    data_file = "heatpoints.js" if is_heatmap else "venues.js"
    data_marker = "HEAT_POINTS" if is_heatmap else "VENUE_DATA"
    try:
        r = requests.get(f"https://{domain}/{data_file}", timeout=30, allow_redirects=True)
        if r.status_code == 200:
            content = r.text
            if data_marker in content:
                expected_sha = manifest.get("sha", "")
                if expected_sha != "unknown" and expected_sha in content:
                    log.info("  [PASS] %s contains expected sha (%s)", data_file, expected_sha)
                elif expected_sha == "unknown":
                    log.info("  [PASS] %s loads (sha check skipped)", data_file)
                else:
                    log.warning("  [WARN] %s sha mismatch (may be stale)", data_file)
                passed += 1

                min_size = 500 if is_heatmap else 1000
                if len(content) > min_size:
                    log.info("  [PASS] %s has data (%d bytes)", data_file, len(content))
                    passed += 1
                else:
                    log.error("  [FAIL] %s too small (%d bytes)", data_file, len(content))
                    failed += 1
            else:
                log.error("  [FAIL] %s missing %s", data_file, data_marker)
                failed += 1
        else:
            log.error("  [FAIL] %s: status=%d", data_file, r.status_code)
            failed += 1
    except Exception as e:
        log.error("  [FAIL] %s: %s", data_file, e)
        failed += 1

    # Test 3: style.css loads
    try:
        r = requests.get(f"https://{domain}/style.css", timeout=10, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 100:
            log.info("  [PASS] style.css loads (%d bytes)", len(r.content))
            passed += 1
        else:
            log.error("  [FAIL] style.css: status=%d, size=%d", r.status_code, len(r.content))
            failed += 1
    except Exception as e:
        log.error("  [FAIL] style.css: %s", e)
        failed += 1

    return passed, failed


def perf_smoke_test_dash(domain: str = "dash.nyc") -> bool:
    """Performance smoke test for the dash app — ensures fast page load.

    1. Structural: early-fetch <script> must appear BEFORE any <style> in HTML
       (Safari parses sequentially; if CSS is first, fetch is delayed ~2.5s)
    2. Timing: headless Chromium with CPU throttling must see data ready < 2s
    """
    import re

    all_ok = True

    # ── Structural check ──
    try:
        r = requests.get(f"https://{domain}/", timeout=15, allow_redirects=True,
                         headers={"Accept-Encoding": "identity"})
        html = r.text

        # Find positions of first <script> containing fetch() and first <style>
        fetch_match = re.search(r"<script[^>]*>.*?fetch\(", html, re.DOTALL)
        style_match = re.search(r"<style", html)

        if fetch_match and style_match:
            if fetch_match.start() < style_match.start():
                log.info("  [PASS] early-fetch <script> appears before first <style> (pos %d < %d)",
                         fetch_match.start(), style_match.start())
            else:
                log.error("  [FAIL] early-fetch <script> (pos %d) appears AFTER first <style> (pos %d) — Safari will delay fetch",
                          fetch_match.start(), style_match.start())
                all_ok = False
        elif not fetch_match:
            log.error("  [FAIL] no <script> with fetch() found in HTML")
            all_ok = False
        else:
            log.info("  [PASS] early-fetch present, no <style> found (edge case)")
    except Exception as e:
        log.error("  [FAIL] structural check: %s", e)
        all_ok = False

    # ── Timing check (headless Chromium, 4x CPU throttle) ──
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            cdp = context.new_cdp_session(page)
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": 4})

            page.goto(f"https://{domain}/", wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(500)

            perf = page.evaluate("""() => {
                const nav = performance.getEntriesByType('navigation')[0];
                const resources = performance.getEntriesByType('resource');
                const marks = performance.getEntriesByType('mark');
                const fetches = resources.filter(r =>
                    r.name.includes('venues.json') || r.name.includes('venue_info.json')
                );
                const markersRendered = marks.find(m => m.name === 'markers-rendered');
                return {
                    responseEnd: nav.responseEnd,
                    loadEvent: nav.loadEventEnd,
                    fetches: fetches.map(f => ({
                        name: f.name.split('/').pop().split('?')[0],
                        startTime: f.startTime,
                        responseEnd: f.responseEnd,
                    })),
                    markersRendered: markersRendered ? markersRendered.startTime : null,
                };
            }""")

            cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
            cdp.detach()
            context.close()
            browser.close()

        if perf["fetches"]:
            earliest = min(f["startTime"] for f in perf["fetches"])
            gap = earliest - perf["responseEnd"]
            latest_end = max(f["responseEnd"] for f in perf["fetches"])

            if gap <= 500:
                log.info("  [PASS] fetch gap: %+.0fms (doc response → first fetch)", gap)
            else:
                log.error("  [FAIL] fetch gap: %+.0fms exceeds 500ms threshold", gap)
                all_ok = False

            if latest_end <= 2000:
                log.info("  [PASS] data ready at %.0fms (threshold: 2000ms)", latest_end)
            else:
                log.error("  [FAIL] data ready at %.0fms exceeds 2000ms threshold", latest_end)
                all_ok = False
        else:
            log.error("  [FAIL] no data fetches detected in timing test")
            all_ok = False

        if perf.get("markersRendered") is not None:
            t = perf["markersRendered"]
            if t <= 2000:
                log.info("  [PASS] markers rendered at %.0fms (threshold: 2000ms)", t)
            else:
                log.error("  [FAIL] markers rendered at %.0fms exceeds 2000ms threshold", t)
                all_ok = False

    except ImportError:
        log.warning("  [SKIP] playwright not installed — skipping timing test")
    except Exception as e:
        log.error("  [FAIL] timing test: %s", e)
        all_ok = False

    return all_ok


def landing_news_smoke_test(domain: str = "dash.nyc") -> bool:
    """Verify the dash landing only shows positive news and venue info matches.

    1. Structural: every venue shown on landing (sentiment=POSITIVE, score>=0.90)
       must have matching venue_info at its original index.
    2. Behavioural (Playwright): click each visible marker, verify the info panel
       never shows 'negative' in the MOOD field.
    """
    all_ok = True

    # ── Structural: verify venue_info alignment via JSON ──
    try:
        venues_r = requests.get(f"https://{domain}/news/venues.json", timeout=15)
        info_r = requests.get(f"https://{domain}/news/venue_info.json", timeout=15)
        venues_data = venues_r.json()
        info_panels = info_r.json()
        all_venues = venues_data.get("venues", [])

        landing = [
            (i, v) for i, v in enumerate(all_venues)
            if v.get("lat") is not None
            and v.get("lng") is not None
            and v.get("meta", {}).get("sentiment") == "POSITIVE"
            and v.get("meta", {}).get("score", 0) >= 0.90
        ]
        log.info("  Landing filter: %d positive of %d total venues", len(landing), len(all_venues))

        mismatched = 0
        for orig_idx, v in landing:
            if orig_idx >= len(info_panels):
                log.error("  [FAIL] venue %d (%s) has no venue_info panel (only %d panels)",
                          orig_idx, v.get("name", "")[:50], len(info_panels))
                mismatched += 1
                continue
            panel = info_panels[orig_idx]
            vname_slug = v.get("name", "")[:30].lower()
            if vname_slug and vname_slug[:10] not in panel.lower():
                log.warning("  [WARN] venue %d name mismatch: '%s' not found in panel",
                            orig_idx, v.get("name", "")[:50])
            # Check that the panel does NOT say "negative"
            if "negative" in panel.lower().split("mood")[1][:30] if "mood" in panel.lower() else "":
                log.error("  [FAIL] venue %d (%s) panel shows negative mood",
                          orig_idx, v.get("name", "")[:50])
                mismatched += 1

        if mismatched == 0:
            log.info("  [PASS] all %d landing venues have correct positive panels", len(landing))
        else:
            log.error("  [FAIL] %d landing venues have mismatched/negative panels", mismatched)
            all_ok = False

    except Exception as e:
        log.error("  [FAIL] landing news structural check: %s", e)
        all_ok = False

    # ── Behavioural: Playwright clicks markers, checks info panels ──
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(f"https://{domain}/", wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(1000)

            # Check that no NEGATIVE venues are in the rendered set
            neg_count = page.evaluate("""() => {
                if (!window.allVenuesRaw) return -1;
                var neg = 0;
                for (var i = 0; i < allVenuesRaw.length; i++) {
                    if (allVenuesRaw[i].meta && allVenuesRaw[i].meta.sentiment === 'NEGATIVE') neg++;
                }
                return neg;
            }""")

            if neg_count == 0:
                log.info("  [PASS] landing allVenuesRaw contains 0 negative articles")
            elif neg_count == -1:
                log.warning("  [SKIP] allVenuesRaw not accessible — page may not have loaded")
            else:
                log.error("  [FAIL] landing allVenuesRaw contains %d negative articles", neg_count)
                all_ok = False

            # Click a few markers and verify the info panel never shows negative mood
            neg_panels = page.evaluate("""() => {
                if (!window.allVenuesRaw || !window.venueInfo) return [];
                var bad = [];
                for (var i = 0; i < allVenuesRaw.length; i++) {
                    var v = allVenuesRaw[i];
                    var panel = venueInfo[v._vi] || '';
                    if (panel.toLowerCase().indexOf('negative') !== -1) {
                        bad.push({name: v.name, vi: v._vi, panel_excerpt: panel.substring(0, 200)});
                    }
                }
                return bad;
            }""")

            if not neg_panels:
                log.info("  [PASS] no landing venue panels contain 'negative' mood")
            else:
                for np in neg_panels[:3]:
                    log.error("  [FAIL] venue '%s' (panel %d) shows negative mood: %s",
                              np["name"][:50], np["vi"], np["panel_excerpt"][:100])
                all_ok = False

            browser.close()

    except ImportError:
        log.warning("  [SKIP] playwright not installed — skipping behavioural test")
    except Exception as e:
        log.error("  [FAIL] landing news behavioural test: %s", e)
        all_ok = False

    return all_ok


def smoke_test(app_id: str) -> bool:
    """Run smoke tests against a deployed app's production domains."""
    dist_dir = DIST / app_id
    manifest_file = dist_dir / "manifest.json"

    if not manifest_file.exists():
        log.error("No manifest.json for app %s", app_id)
        return False

    manifest = json.loads(manifest_file.read_text())
    domains = manifest.get("domains", [])
    if not domains:
        log.info("No production domains for app %s — tested via sub-path routing", app_id)
        return True

    all_pass = True
    for domain in domains:
        log.info("Smoke testing https://%s ...", domain)
        p, f = smoke_test_domain(domain, manifest)
        log.info("  Results for %s: %d passed, %d failed", domain, p, f)
        if f > 0:
            all_pass = False
    return all_pass


def smoke_test_staging() -> bool:
    """Run smoke tests against staging.dash.nyc."""
    all_ok = True

    # Test the main dash app at staging.dash.nyc
    manifest_path = DIST / "dash" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        log.info("Smoke testing staging: https://staging.dash.nyc ...")
        p, f = smoke_test_domain("staging.dash.nyc", manifest)
        log.info("  Results for staging.dash.nyc: %d passed, %d failed", p, f)
        if f > 0:
            all_ok = False

        # Test sub-apps via staging paths
        for sub in ("eat", "drink", "food", "news", "art", "shop", "theater", "play", "movies", "music", "make", "relax", "pee", "weather", "today", "vibe"):
            sub_manifest_path = DIST / sub / "manifest.json"
            if sub_manifest_path.exists():
                sub_manifest = json.loads(sub_manifest_path.read_text())
                log.info("Smoke testing staging: https://staging.dash.nyc/%s ...", sub)
                p, f = smoke_test_domain(f"staging.dash.nyc/{sub}", sub_manifest)
                log.info("  Results for staging.dash.nyc/%s: %d passed, %d failed", sub, p, f)
                if f > 0:
                    all_ok = False

    # Performance test for dash on staging
    log.info("Running perf smoke test on staging.dash.nyc ...")
    if not perf_smoke_test_dash("staging.dash.nyc"):
        all_ok = False

    # Landing news test on staging
    log.info("Running landing news smoke test on staging.dash.nyc ...")
    if not landing_news_smoke_test("staging.dash.nyc"):
        all_ok = False

    return all_ok


def smoke_test_production() -> bool:
    """Run smoke tests against all production domains."""
    app_ids = sorted(
        d.name for d in DIST.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )
    all_ok = True
    for app_id in app_ids:
        if not smoke_test(app_id):
            all_ok = False

    # Performance test for dash on production
    log.info("Running perf smoke test on dash.nyc ...")
    if not perf_smoke_test_dash("dash.nyc"):
        all_ok = False

    # Landing news test on production
    log.info("Running landing news smoke test on dash.nyc ...")
    if not landing_news_smoke_test("dash.nyc"):
        all_ok = False

    return all_ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NYC Map — deploy with staging")
    parser.add_argument("apps", nargs="*", help="App IDs to deploy (or --all)")
    parser.add_argument("--all", action="store_true", help="Deploy all built apps")
    parser.add_argument("--dry-run", action="store_true", help="rsync dry run")
    parser.add_argument("--smoke-only", action="store_true", help="Skip deploy, run smoke tests")
    parser.add_argument("--promote-automatically", action="store_true",
                        help="Auto-promote staging → production if smoke tests pass")
    parser.add_argument("--promote", action="store_true",
                        help="Promote current staging to production")
    parser.add_argument("--rollback", action="store_true",
                        help="Roll production back to previous release")
    parser.add_argument("--status", action="store_true",
                        help="Show current staging/production release info")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove old releases (keep 5 most recent)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle operational commands
    if args.status:
        show_status()
        return

    if args.promote:
        ok = promote()
        if not ok:
            sys.exit(1)
        log.info("Running production smoke tests...")
        if not smoke_test_production():
            log.error("Production smoke tests failed after promotion!")
            sys.exit(1)
        log.info("Production smoke tests passed.")
        return

    if args.rollback:
        ok = rollback()
        if not ok:
            sys.exit(1)
        return

    if args.cleanup:
        cleanup_releases()
        return

    # Smoke-only mode: test production
    if args.smoke_only:
        app_ids = sorted(
            d.name for d in DIST.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ) if args.all else (args.apps or [])
        if not app_ids:
            parser.error("Specify app(s) or use --all")
        all_ok = True
        for app_id in app_ids:
            if not smoke_test(app_id):
                all_ok = False
        if not all_ok:
            sys.exit(1)
        log.info("All smoke tests passed.")
        return

    # ── Normal deploy flow ──
    # rsync to releases/<sha>/ → point staging symlink → smoke test staging
    # → optionally promote to production
    sha = get_build_sha()
    log.info("Release SHA: %s", sha)

    ok = deploy_release(sha, dry_run=args.dry_run)
    if not ok:
        log.error("Deployment failed!")
        sys.exit(1)

    if args.dry_run:
        return

    # Smoke tests against staging
    log.info("Running staging smoke tests against staging.dash.nyc ...")
    staging_ok = smoke_test_staging()
    if not staging_ok:
        log.error("Staging smoke tests FAILED — release %s NOT promoted.", sha)
        log.error("Fix issues and re-deploy, or manually: python deploy.py --promote -v")
        sys.exit(1)

    log.info("Staging smoke tests passed for release %s", sha)

    if args.promote_automatically:
        log.info("Auto-promoting to production...")
        ok = promote()
        if not ok:
            log.error("Promotion failed!")
            sys.exit(1)
        log.info("Running production smoke tests...")
        if not smoke_test_production():
            log.error("Production smoke tests failed! Consider: python deploy.py --rollback -v")
            sys.exit(1)
        log.info("Release %s is live in production.", sha)
    else:
        log.info("Staging is ready at https://staging.dash.nyc")
        log.info("To promote to production run:  python deploy.py --promote -v")

    # Clean up old releases
    cleanup_releases()


if __name__ == "__main__":
    main()

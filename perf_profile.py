#!/usr/bin/env python3
"""Profile dash.nyc page load with CPU throttling to reproduce real Safari delays.

Uses CDP (Chrome DevTools Protocol) to apply CPU throttling so we can reproduce
the ~2.5s JS parse delay seen in real Safari on a Mac.

Usage:
    python3 perf_profile.py                  # default: 4x CPU slowdown
    python3 perf_profile.py --cpu=6          # 6x CPU slowdown
    python3 perf_profile.py --no-throttle    # no throttling
    python3 perf_profile.py https://staging.dash.nyc
"""

import sys
import time
from playwright.sync_api import sync_playwright


def profile_url(url: str, runs: int = 3, cpu_throttle: float = 4.0) -> list[dict]:
    """Load URL in headless Chromium with CPU throttling, extract timings."""
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for i in range(runs):
            context = browser.new_context()
            page = context.new_page()

            # Apply CPU throttling via CDP
            cdp = context.new_cdp_session(page)
            if cpu_throttle > 1.0:
                cdp.send("Emulation.setCPUThrottlingRate", {"rate": cpu_throttle})

            # Navigate and wait for load + network idle
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1000)

            # Extract everything via Performance API
            perf = page.evaluate("""() => {
                const nav = performance.getEntriesByType('navigation')[0];
                const resources = performance.getEntriesByType('resource');
                const marks = performance.getEntriesByType('mark');

                const fetches = resources.filter(r =>
                    r.name.includes('venues.json') || r.name.includes('venue_info.json')
                );

                return {
                    responseEnd: nav.responseEnd,
                    domInteractive: nav.domInteractive,
                    domContentLoaded: nav.domContentLoadedEventEnd,
                    loadEvent: nav.loadEventEnd,
                    transferSize: nav.transferSize,
                    decodedBodySize: nav.decodedBodySize,
                    marks: marks.map(m => ({ name: m.name, startTime: m.startTime })),
                    fetches: fetches.map(f => ({
                        name: f.name.split('/').pop().split('?')[0],
                        startTime: f.startTime,
                        responseEnd: f.responseEnd,
                        transferSize: f.transferSize,
                    })),
                    allResources: resources.map(r => ({
                        name: r.name.split('/').pop().split('?')[0],
                        type: r.initiatorType,
                        startTime: r.startTime,
                        responseEnd: r.responseEnd,
                        transferSize: r.transferSize,
                    })),
                };
            }""")

            if cpu_throttle > 1.0:
                cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
            cdp.detach()

            results.append(perf)
            context.close()

        browser.close()

    return results


def print_report(results: list, url: str, cpu_throttle: float):
    print(f"\n{'='*70}")
    print(f"  Performance Profile: {url}")
    print(f"  CPU Throttle: {cpu_throttle}x | Runs: {len(results)}")
    print(f"{'='*70}\n")

    all_markers_rendered = []

    for i, perf in enumerate(results):
        print(f"--- Run {i+1} ---")
        print(f"  Document:  {perf['transferSize']:,}b xfer, {perf['decodedBodySize']:,}b decoded")
        print(f"  Response complete:     {perf['responseEnd']:>7.0f} ms")
        print(f"  DOM Interactive:       {perf['domInteractive']:>7.0f} ms")
        print(f"  DOM Content Loaded:    {perf['domContentLoaded']:>7.0f} ms")
        print(f"  Load Event:            {perf['loadEvent']:>7.0f} ms")

        if perf['marks']:
            print(f"\n  Performance marks:")
            for m in sorted(perf['marks'], key=lambda x: x['startTime']):
                print(f"    {m['startTime']:>7.0f} ms  {m['name']}")
                if m['name'] == 'markers-rendered':
                    all_markers_rendered.append(m['startTime'])

        if perf['fetches']:
            earliest_fetch_start = min(f['startTime'] for f in perf['fetches'])
            latest_fetch_end = max(f['responseEnd'] for f in perf['fetches'])
            gap = earliest_fetch_start - perf['responseEnd']

            print(f"\n  Data fetches:")
            for f in sorted(perf['fetches'], key=lambda x: x['startTime']):
                print(f"    {f['name']:25s}  start={f['startTime']:>7.0f}ms  end={f['responseEnd']:>7.0f}ms  {f['transferSize']:>6,}b")

            print(f"\n  ** GAP (doc response → first fetch): {gap:>+.0f} ms **")
            print(f"  ** Data ready at: {latest_fetch_end:.0f} ms **")
        else:
            print(f"\n  WARNING: No venues.json/venue_info.json fetches detected!")

        print()

    if len(results) > 1:
        gaps = []
        data_ready = []
        for perf in results:
            if perf['fetches']:
                earliest = min(f['startTime'] for f in perf['fetches'])
                latest = max(f['responseEnd'] for f in perf['fetches'])
                gaps.append(earliest - perf['responseEnd'])
                data_ready.append(latest)

        print(f"{'='*70}")
        print(f"  SUMMARY ({len(results)} runs, {cpu_throttle}x CPU throttle)")
        if gaps:
            print(f"  GAP (doc->fetch):  avg={sum(gaps)/len(gaps):+.0f}ms  min={min(gaps):+.0f}ms  max={max(gaps):+.0f}ms")
            print(f"  Data ready:        avg={sum(data_ready)/len(data_ready):.0f}ms  min={min(data_ready):.0f}ms  max={max(data_ready):.0f}ms")
        if all_markers_rendered:
            print(f"  Markers rendered:  avg={sum(all_markers_rendered)/len(all_markers_rendered):.0f}ms  min={min(all_markers_rendered):.0f}ms  max={max(all_markers_rendered):.0f}ms")
        print(f"{'='*70}")

    return all_markers_rendered


if __name__ == "__main__":
    url = "https://dash.nyc"
    cpu_throttle = 4.0
    runs = 3

    for arg in sys.argv[1:]:
        if arg.startswith("http"):
            url = arg
        elif arg.startswith("--cpu="):
            cpu_throttle = float(arg.split("=")[1])
        elif arg == "--no-throttle":
            cpu_throttle = 1.0
        elif arg.startswith("--runs="):
            runs = int(arg.split("=")[1])

    print(f"Profiling {url} with {cpu_throttle}x CPU throttle, {runs} runs...")
    results = profile_url(url, runs=runs, cpu_throttle=cpu_throttle)
    print_report(results, url, cpu_throttle)

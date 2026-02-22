/* ============================================================
   dash-transition.js — Cross-subdomain page transition system
   Uses document.referrer (no cookies) to detect navigation
   between dash.nyc domains and play matching animations.
   ============================================================ */
(function () {
  "use strict";

  /* ── Configuration ──
     Override these on a per-page basis by setting them before
     this script runs, or by adding data- attributes to the
     #dash-transition element.

     Exit animation:  what plays on THIS page when navigating away
     Entry animation: what plays on THIS page when arriving

     Pair convention (default):
       exit-zoom-in   →  entry-zoom-in   (zoom into → zoom through)
       exit-slide-up  →  entry-slide-down (slide up → slide back down)
  */

  var DASH_DOMAINS = ["dash.nyc", "eat.dash.nyc", "drink.dash.nyc", "food.dash.nyc"];

  /* Detect where we came from via referrer */
  var ref = document.referrer || "";
  var fromDash = false;
  var fromDomain = "";
  try {
    if (ref) {
      var u = new URL(ref);
      fromDomain = u.hostname;
      fromDash = DASH_DOMAINS.indexOf(fromDomain) !== -1;
    }
  } catch (e) { /* invalid referrer */ }

  /* Find or create the transition overlay */
  var overlay = document.getElementById("dash-transition");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "dash-transition";
    document.body.appendChild(overlay);
  }

  /* Read config from data attributes or defaults */
  var entryAnim = overlay.dataset.entry || "zoom-in";     /* from another dash domain */
  var coldAnim = overlay.dataset.cold || "fade";           /* from external / direct */
  var exitAnim = overlay.dataset.exit || "zoom-in";        /* when leaving this page */

  /* ── Play entry animation ── */
  function playEntry() {
    var anim = fromDash ? entryAnim : coldAnim;
    overlay.className = "entry-" + anim;
    overlay.addEventListener("animationend", function () {
      overlay.style.display = "none";
      overlay.className = "";
    }, { once: true });
  }

  /* ── Exit animation + redirect ── */
  function navigateWithTransition(href, exitOverride) {
    var ex = exitOverride || exitAnim;
    overlay.style.display = "";
    overlay.className = "exit-" + ex;
    overlay.addEventListener("animationend", function () {
      window.location.href = href;
    }, { once: true });
    /* Safety: redirect even if animation somehow fails */
    var dur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--dash-exit-duration")) || 0.1;
    setTimeout(function () { window.location.href = href; }, dur * 1000 + 50);
  }

  /* Start entry animation immediately */
  playEntry();

  /* ── Public API ── */
  window.dashTransition = {
    navigate: navigateWithTransition,
    fromDash: fromDash,
    fromDomain: fromDomain
  };
})();

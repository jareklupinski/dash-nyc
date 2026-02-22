/* ── Fairy flight sprite (lo-fi) ──────────────────────────────
   Pixelated sparkle that flies from tap to the venue panel
   along a bezier curve with chunky square particles.
   Usage:  fairySprite.track(e)  on pointerdown
           fairySprite.fly()     inside showVenueInfo            */
var fairySprite = (function () {
  var _n = 0, _pt = null;
  var ACCENT;

  function hexToRgb(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  /* snap to pixel grid for that crunchy look */
  function snap(v, g) { return Math.round(v / g) * g; }

  function getTarget() {
    var vi = document.getElementById('venue-info');
    if (vi) {
      var r = vi.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        return [r.left + r.width / 2, r.top + r.height / 2];
      }
    }
    /* fallback */
    var mob = window.innerWidth <= 700;
    return mob ? [window.innerWidth / 2, window.innerHeight] : [160, window.innerHeight / 2];
  }

  function fly() {
    if (!_pt) return;
    if (!ACCENT) ACCENT = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#d4a017';
    var rgb = hexToRgb(ACCENT);
    var baseOp = [1, 0.75, 0.5, 0.25, 0.05][_n < 5 ? _n : 4];
    _n++;

    var sx = _pt[0], sy = _pt[1];
    _pt = null;
    var target = getTarget();
    var ex = target[0], ey = target[1];
    var PX = 3; /* pixel grid size */

    /* bezier arc control points */
    var dx = ex - sx, dy = ey - sy;
    var len = Math.sqrt(dx * dx + dy * dy) || 1;
    var nx = -dy / len, ny = dx / len;
    var mob = window.innerWidth <= 700;
    var bulge = Math.min(len * 0.35, 100) * (mob ? -1 : 1);
    var cx1 = sx + dx * 0.25 + nx * bulge;
    var cy1 = sy + dy * 0.25 + ny * bulge;
    var cx2 = sx + dx * 0.75 + nx * bulge * 0.5;
    var cy2 = sy + dy * 0.75 + ny * bulge * 0.5;

    /* canvas — no DPR scaling for chunky pixels */
    var cvs = document.createElement('canvas');
    cvs.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:10000;image-rendering:pixelated';
    cvs.width = Math.ceil(window.innerWidth / PX);
    cvs.height = Math.ceil(window.innerHeight / PX);
    var ctx = cvs.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    document.body.appendChild(cvs);

    var S = 1 / PX; /* coord scale */

    /* palette: accent + white + gold */
    var cols = [
      [rgb[0], rgb[1], rgb[2]],
      [255, 255, 255],
      [255, 220, 100],
      [Math.min(rgb[0]+60,255), Math.min(rgb[1]+60,255), Math.min(rgb[2]+30,255)],
    ];

    function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a.toFixed(2) + ')'; }

    var particles = [];
    var bursts = [];
    var DUR = 420;
    var start = null;
    var done = false;

    function bezier(t) {
      var u = 1 - t;
      return [
        u*u*u*sx + 3*u*u*t*cx1 + 3*u*t*t*cx2 + t*t*t*ex,
        u*u*u*sy + 3*u*u*t*cy1 + 3*u*t*t*cy2 + t*t*t*ey
      ];
    }

    function spawnTrail(x, y) {
      for (var i = 0; i < 2; i++) {
        var a = Math.random() * Math.PI * 2;
        var spd = 0.3 + Math.random() * 0.8;
        particles.push({
          x: x * S, y: y * S,
          vx: Math.cos(a) * spd, vy: Math.sin(a) * spd + 0.2,
          life: 1, decay: 0.025 + Math.random() * 0.03,
          col: cols[(Math.random() * cols.length) | 0],
          blink: (Math.random() * 6) | 0,
        });
      }
    }

    function spawnBurst(x, y) {
      for (var i = 0; i < 16; i++) {
        var a = (Math.PI * 2 / 16) * i + (Math.random() - 0.5) * 0.4;
        var spd = 1 + Math.random() * 2;
        bursts.push({
          x: x * S, y: y * S,
          vx: Math.cos(a) * spd, vy: Math.sin(a) * spd,
          life: 1, decay: 0.025 + Math.random() * 0.02,
          col: cols[(Math.random() * cols.length) | 0],
          trail: [],
        });
      }
    }

    function frame(ts) {
      if (!start) start = ts;
      var elapsed = ts - start;
      var t = Math.min(elapsed / DUR, 1);
      var et = 1 - Math.pow(1 - t, 3);
      var pos = bezier(et);
      var w = cvs.width, h = cvs.height;

      ctx.clearRect(0, 0, w, h);

      if (t < 1) {
        spawnTrail(pos[0], pos[1]);
      } else if (!done) {
        done = true;
        spawnBurst(pos[0], pos[1]);
      }

      /* trail particles — square pixels, blink on/off */
      for (var i = particles.length - 1; i >= 0; i--) {
        var p = particles[i];
        p.x += p.vx; p.y += p.vy;
        p.vy += 0.05;
        p.vx *= 0.96;
        p.life -= p.decay;
        p.blink++;
        if (p.life <= 0) { particles.splice(i, 1); continue; }
        if (p.blink % 3 === 0) continue; /* blink off every 3rd frame */
        var a = p.life * baseOp;
        ctx.fillStyle = rgba(p.col, a);
        ctx.fillRect(Math.round(p.x), Math.round(p.y), 1, 1);
      }

      /* burst particles with trail */
      for (var j = bursts.length - 1; j >= 0; j--) {
        var b = bursts[j];
        b.trail.push({ x: Math.round(b.x), y: Math.round(b.y), life: b.life });
        if (b.trail.length > 4) b.trail.shift();
        b.x += b.vx; b.y += b.vy;
        b.vy += 0.08;
        b.vx *= 0.95;
        b.life -= b.decay;
        if (b.life <= 0) { bursts.splice(j, 1); continue; }
        /* trail dots */
        for (var k = 0; k < b.trail.length; k++) {
          var tr = b.trail[k];
          ctx.fillStyle = rgba(b.col, (k / b.trail.length) * b.life * baseOp * 0.5);
          ctx.fillRect(tr.x, tr.y, 1, 1);
        }
        /* head */
        ctx.fillStyle = rgba(b.col, b.life * baseOp);
        ctx.fillRect(Math.round(b.x), Math.round(b.y), 1, 1);
      }

      /* fairy head — 2x2 pixel block + cross */
      if (t < 1) {
        var fx = Math.round(pos[0] * S), fy = Math.round(pos[1] * S);
        /* core block */
        ctx.fillStyle = rgba([255,255,255], 0.9 * baseOp);
        ctx.fillRect(fx, fy, 2, 2);
        /* tiny cross arms — blink */
        if (((elapsed / 80) | 0) % 2 === 0) {
          ctx.fillStyle = rgba(cols[0], 0.6 * baseOp);
          ctx.fillRect(fx - 1, fy, 1, 2);
          ctx.fillRect(fx + 2, fy, 1, 2);
          ctx.fillRect(fx, fy - 1, 2, 1);
          ctx.fillRect(fx, fy + 2, 2, 1);
        }
      }

      if (t < 1 || particles.length > 0 || bursts.length > 0) {
        requestAnimationFrame(frame);
      } else {
        cvs.remove();
      }
    }
    requestAnimationFrame(frame);
  }

  return {
    track: function (e) { _pt = [e.clientX, e.clientY]; },
    fly: fly
  };
})();

/* The walk is a state machine, not a scroll scrub: scroll decides which stop
   you are on, and CSS plays the move between stops on its own clock. */
(function () {
  "use strict";

  var READINGS = [
    { name: "Conventional RGB",        spec: "40 ms exposure \u00b7 motion blur" },
    { name: "IMU",                     spec: "six mounts \u00b7 bias & drift" },
    { name: "Direct ToF LiDAR",        spec: "range, coloured by depth" },
    { name: "Thermal camera",          spec: "heat solve \u00b7 kelvin \u00b7 ironbow" },
    { name: "Indirect time-of-flight", spec: "lock-in taps \u00b7 20 MHz" },
    { name: "Event camera",            spec: "1.6 kHz \u00b7 ON / OFF polarity" },
    { name: "Single-photon camera",    spec: "1.6 kHz \u00b7 Bernoulli detection" }
  ];

  var N = READINGS.length;      /* 7 sensors */
  var SETTLE = N;               /* stop 7: the whole pile, settled, no result claimed */
  var BAND = 1 / (N + 1);       /* one band per stop, the pile included */
  var SETTLE_AT = N * BAND;     /* the last band is the whole pile, held as a stop */

  var SETTLE_NOTE = {
    name: "Multi-sensor",
    spec: "one world, many measurements"
  };

  var deckSec = document.querySelector(".hero");
  var deck = document.querySelector(".deck");
  var layers = [].slice.call(document.querySelectorAll(".deck__layer"));
  var railBtns = [].slice.call(document.querySelectorAll(".rail button"));
  var nav = document.getElementById("nav");
  var idxEl = document.querySelector("[data-index]");
  var nameEl = document.querySelector("[data-name]");
  var specEl = document.querySelector("[data-spec]");

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var clamp = function (v, a, b) { return v < a ? a : v > b ? b : v; };

  function videoOf(el) { return el ? el.querySelector("video") : null; }

  function play(v) {
    if (!v) return;
    if (v.preload === "none") v.preload = "auto";
    v.dataset.on = "1";
    var pr = v.play();
    if (pr && pr.catch) pr.catch(function () {});
  }

  function pause(v) {
    if (!v || !v.dataset.on) return;
    delete v.dataset.on;
    v.pause();
  }

  /* only the reading on screen runs; the rest wait on their poster */
  function onlyPlaying(index) {
    layers.forEach(function (el, k) {
      var v = videoOf(el);
      if (!v) return;
      if (k === index) play(v); else pause(v);
    });
  }

  var state = -1;

  function applyState(s) {
    if (s === state) return;
    state = s;
    var settled = s === SETTLE;

    var front = Math.min(s, N - 1);
    var spread = s >= 1 ? 1 : 0;                 /* flat for the head-on opener only */
    var grow = window.innerWidth <= 900 ? 1.05 : 1.25;   /* phones cannot fan as wide */
    if (!settled) grow = 1;
    var du = parseFloat(getComputedStyle(deck).getPropertyValue("--du")) || 60;

    layers.forEach(function (el, i) {
      var landed = i <= front;
      var back = front - i;
      el.style.setProperty("--pr", landed ? "1" : "0");
      /* deep plates queue in front of the pile, so an arriving reading flies in */
      var z = landed ? (settled ? (i - (N - 1) / 2) * du * grow
                              : -back * du * spread) : 210;
      el.style.setProperty("--zu", String(z));
      /* opaque or gone: a half-transparent plate reads as milk over the pile */
      el.style.setProperty("--dim", landed && (settled || back <= 4) ? "1" : "0");
    });

    if (deck) deck.style.setProperty("--up", s >= 1 ? "1" : "0");

    if (idxEl) idxEl.textContent = settled ? "\u2014" : "0" + (s + 1);
    if (nameEl) nameEl.textContent = settled ? SETTLE_NOTE.name : READINGS[s].name;
    if (specEl) specEl.textContent = settled ? SETTLE_NOTE.spec : READINGS[s].spec;

    railBtns.forEach(function (b, bi) {
      /* the pile is a stop in its own right, so it lights the list too */
      if (bi === (settled ? SETTLE : s)) b.setAttribute("aria-current", "true");
      else b.removeAttribute("aria-current");
    });

    onlyPlaying(settled ? -1 : s);
  }

  function geometry() {
    var r = deckSec.getBoundingClientRect();
    return { top: window.scrollY + r.top, span: Math.max(1, r.height - window.innerHeight) };
  }

  function stateFor(prog) {
    /* each stop owns one band, so where you park inside a band is the stop you
       get — hover mid-band, on the edge, it does not matter */
    return clamp(Math.floor(prog / BAND), 0, SETTLE);
  }

  function scrollForState(s, g) {
    var t = s === SETTLE ? SETTLE_AT + BAND * 0.5 : (s + 0.5) * BAND;
    return g.top + clamp(t, 0, 1) * g.span;
  }

  var ticking = false;
  var lastChange = 0;
  var jumping = false;
  var lockUntil = 0, lockAt = 0;
  var ready = false;   /* no fling guard while the page is restoring scroll */
  var lastScrollAt = 0;  /* a scrollbar click is one event; a fling is many */

  function onScroll() {
    if (ticking || !deckSec) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var g = geometry();
      var prog = clamp((window.scrollY - g.top) / g.span, 0, 1);
      var next = stateFor(prog);

      /* one gesture advances one stop: a phone fling that would carry you past
         two readings is pulled back to the stop you were headed for */
      var now = Date.now();
      var flinging = now - lastScrollAt < 120;   /* continuous, so momentum not a jump */
      lastScrollAt = now;

      /* one skipped stop is momentum; three or more is a deliberate jump */
      if (ready && flinging && !jumping && state >= 0 && Math.abs(next - state) === 2) {
        /* hold the page at the next stop while the fling burns itself out */
        next = state + Math.sign(next - state);
        lockAt = scrollForState(next, g);
        lockUntil = Date.now() + 700;
      }
      if (Date.now() < lockUntil) {
        if (Math.abs(window.scrollY - lockAt) > 4) window.scrollTo(0, lockAt);
        prog = clamp((lockAt - g.top) / g.span, 0, 1);
        next = stateFor(prog);
      }

      if (next !== state) lastChange = Date.now();
      applyState(next);
      if (nav) nav.classList.toggle("is-stuck", window.scrollY > 80);
    });
  }

  if (deckSec && deck) {
    if (reduceMotion) {
      applyState(N - 1);        /* the pile sits open, nothing moves */
    } else {
      applyState(stateFor(0));
      setTimeout(function () { ready = true; }, 350);
      window.addEventListener("scroll", onScroll, { passive: true });
      window.addEventListener("resize", onScroll, { passive: true });
      onScroll();
    }
  }

  railBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var i = parseInt(btn.getAttribute("data-goto"), 10);
      if (!deckSec) return;
      var g = geometry();
      var target = (i + 0.5) * BAND;
      jumping = true;                        /* a jump is meant to cross stops */
      setTimeout(function () { jumping = false; }, 900);
      window.scrollTo({
        top: g.top + target * g.span + 1,
        behavior: reduceMotion ? "auto" : "smooth"
      });
    });
  });

  var gridVideos = [].slice.call(document.querySelectorAll(".layers-grid video, .dataset video"));
  if ("IntersectionObserver" in window && gridVideos.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) play(v); else pause(v);
      });
    }, { threshold: 0.25 });
    gridVideos.forEach(function (v) { io.observe(v); });
  }

  var onNav = function () {
    if (nav) nav.classList.toggle("is-stuck", window.scrollY > 80);
  };
  window.addEventListener("scroll", onNav, { passive: true });
  onNav();
})();

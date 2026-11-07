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

  /* ---- clips ------------------------------------------------------------
     A poster image rides over every clip and only steps aside while frames
     are really arriving, so a refused, stalled or unloaded decode shows the
     frame instead of a black square. */
  function syncLive(v) {
    var clip = v.closest(".clip");
    if (clip) {
      /* a scrubbed clip never plays, so its frames count as live ones */
      var live = v.dataset.walk ? v.readyState >= 2 : (!v.paused && v.readyState >= 2);
      clip.classList.toggle("is-live", live);
    }
  }

  function setClip(v, on) {
    if (!v) return;
    if (on) {
      if (v.preload === "none") v.preload = "auto";
      v.dataset.on = "1";
      var pr = v.play();
      if (pr && pr.catch) pr.catch(function () {});
    } else if (v.dataset.on) {
      delete v.dataset.on;
      v.pause();
    }
    syncLive(v);
  }

  ["playing", "timeupdate", "seeked", "loadeddata", "canplay",
   "pause", "waiting", "stalled", "error", "emptied", "ended"].forEach(function (type) {
    [].forEach.call(document.querySelectorAll(".clip > video"), function (v) {
      v.addEventListener(type, function () { syncLive(v); });
    });
  });

  /* only the reading on screen runs, and only while the hero is in view */
  var heroOnScreen = true;
  function onlyPlaying(index) {
    layers.forEach(function (el, k) {
      setClip(videoOf(el), k === index && heroOnScreen);
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

  if (deckSec && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      var vis = entries[0].isIntersecting;
      if (vis === heroOnScreen) return;
      heroOnScreen = vis;
      if (state >= 0) onlyPlaying(state === SETTLE ? -1 : state);
    }, {}).observe(deckSec);
  }

  var datasetVideo = document.querySelector(".dataset video");
  if (datasetVideo && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      setClip(datasetVideo, entries[0].isIntersecting);
    }, { threshold: 0.25 }).observe(datasetVideo);
  }

  /* ---- the annotation grid is one shot read fourteen ways --------------
     All fourteen clips are the same 9.01 s camera move, so they are not a
     slideshow of independent loops: they are pulled down a few at a time,
     and only when the last one is buffered does the whole grid start, every
     tile from zero. Until that moment each tile sits on its poster. */
  var ensemble = [].slice.call(document.querySelectorAll(".layers-grid video"));
  var gridReady = false, gridInView = false, gridRunning = false, drift = null;

  /* all fourteen clips are the same 9.01 s camera move, so the grid is one
     clock: nothing moves until the last clip is buffered, then every tile
     starts from zero and is nudged back if its decoder drifts */
  function syncGrid() {
    var run = gridReady && gridInView;
    if (run === gridRunning) return;
    gridRunning = run;
    clearInterval(drift);
    ensemble.forEach(function (v) { setClip(v, run); });
    if (!run) return;
    ensemble.forEach(function (v) { try { v.currentTime = 0; } catch (e) {} });
    ensemble.forEach(function (v) { setClip(v, true); });
    drift = setInterval(function () {
      var t = ensemble[0].currentTime;
      ensemble.forEach(function (v) {
        if (v === ensemble[0]) return;
        if (v.paused) setClip(v, true);
        else if (Math.abs(v.currentTime - t) > 0.18) { try { v.currentTime = t; } catch (e) {} }
      });
    }, 1200);
  }

  /* buffered a few at a time, so the posters win the first of the bandwidth */
  function preloadAll(list, done) {
    var i = 0, live = 0, finished = false;
    function step() {
      while (live < 3 && i < list.length) {
        (function (v) {
          live++;
          function settle() {
            v.removeEventListener("canplaythrough", settle);
            v.removeEventListener("loadeddata", settle);
            v.removeEventListener("error", settle);
            live--; step();
          }
          v.addEventListener("canplaythrough", settle);
          v.addEventListener("loadeddata", settle);
          v.addEventListener("error", settle);
          v.preload = "auto";
          v.load();
        })(list[i++]);
      }
      if (!finished && i >= list.length && live === 0) { finished = true; done(); }
    }
    step();
  }

  var gridEl = document.querySelector(".layers-grid");
  if (gridEl && ensemble.length && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      if (!entries[0].isIntersecting) return;
      preloadAll(ensemble.slice(), function () { gridReady = true; syncGrid(); });
    }, { rootMargin: "600px 0px", threshold: 0 }).observe(gridEl);

    new IntersectionObserver(function (entries) {
      gridInView = entries[0].isIntersecting;
      syncGrid();
    }, { threshold: [0, 0.15] }).observe(gridEl);
  }

  /* ---- the readings walk ------------------------------------------------
     A player, not a scroll scrub. Each reading declares how long it wants to
     sit on each of its beats; the player turns that into a step timeline, so
     the meter is not decoration — a segment reaching 100% is what starts the
     next reading. The four cycle forever; arrows and swipes step by hand. */
  var walkEl = document.querySelector(".walk");
  var walkArmed = !!walkEl && !reduceMotion;

  function startWalk() {
    var readings = [].slice.call(walkEl.querySelectorAll(".reading"));
    var videos = readings.map(function (r) { return r.querySelector("video"); });
    var segs = [].slice.call(walkEl.querySelectorAll(".walk__meter span"));
    var arrows = [].slice.call(walkEl.querySelectorAll(".walk__arrow"));
    var toggleEl = walkEl.querySelector(".walk__toggle");
    var statusEl = walkEl.querySelector(".walk__status");
    var N = readings.length;
    var current = -1, timers = [], frame = 0, playing = false;

    var plan = readings.map(function (r) {
      var beats = (r.getAttribute("data-beats") || "").split(",").filter(Boolean).map(function (b) {
        var p = b.split(":");
        return { t: parseFloat(p[0]), inset: parseInt(p[1], 10), dwell: parseInt(p[2] || "0", 10) };
      });
      return {
        beats: beats,
        loop: r.hasAttribute("data-loop"),
        reveal: parseFloat(r.getAttribute("data-reveal")) || 0,
        ms: parseInt(r.getAttribute("data-ms") || "0", 10),
        passes: parseInt(r.getAttribute("data-passes") || "2", 10)
      };
    });

    /* how long the row's copy takes to read, so the sequence never swaps out
       from under someone reading it. ~4 words a second plus a beat to arrive. */
    function readingMs(i) {
      var p = readings[i].querySelector(".reading__copy p");
      var words = p ? p.textContent.trim().split(/\s+/).length : 0;
      return Math.round(words * 250) + 900;
    }

    /* a reading becomes a list of steps: play to a beat, sit on it, play on */
    function stepsFor(i) {
      var p = plan[i], out = [];
      if (p.loop) { out.push({ kind: "loop", from: 0, to: null, reveal: p.reveal, ms: p.ms || 6000, passes: p.passes }); return out; }
      if (!p.beats.length) { out.push({ kind: "hold", inset: 0, ms: p.ms || 4000 }); return out; }
      out.push({ kind: "play", from: 0, to: p.beats[0].t });
      p.beats.forEach(function (b, k) {
        out.push({ kind: "hold", inset: b.inset, at: b.t, ms: b.dwell || 2800 });
        var next = p.beats[k + 1];
        if (next) out.push({ kind: "play", from: b.t, to: next.t });
      });
      /* if the media finishes before the copy has been read, the last step
         keeps going, so the meter and the hand-over stay one clock */
      var need = readingMs(i);
      var have = out.reduce(function (sum, s) { return sum + stepMs(s); }, 0);
      if (have < need) {
        var last = out[out.length - 1];
        if (last.kind === "loop") last.passes = Math.ceil(need / last.ms) * last.passes;
        else last.ms += need - have;
      }
      return out;
    }

    var steps = [], stepAt = 0, stepStart = 0, total = 1;

    function stepMs(s) {
      if (s.kind === "play") return Math.max(1, (s.to - s.from) * 1000);
      return s.ms;
    }

    function clearTimers() { timers.forEach(function (t) { clearTimeout(t.id); }); timers = []; }
    /* a timer leaves the list the moment it fires, or a later resume would
       re-arm it with no time left and jump the sequence forward */
    function later(fn, ms) {
      var rec = { remaining: ms, due: Date.now() + ms, id: 0 };
      rec.run = function () {
        var at = timers.indexOf(rec);
        if (at >= 0) timers.splice(at, 1);
        fn();
      };
      rec.id = setTimeout(rec.run, ms);
      timers.push(rec);
    }
    function armTimers() {
      timers.forEach(function (t) {
        t.due = Date.now() + t.remaining;
        t.id = setTimeout(t.run, t.remaining);
      });
    }
    function disarmTimers() {
      var now = Date.now();
      timers.forEach(function (t) { clearTimeout(t.id); t.remaining = Math.max(0, t.due - now); });
    }
    function paint(r, inset) { r.setAttribute("data-beat", String(inset)); r.classList.add("is-held"); }

    /* the stage shows one reading at a time, so the other three leave the
       accessibility tree with it */
    function focusTree(i) {
      readings.forEach(function (r, k) {
        if (k === i) r.removeAttribute("aria-hidden");
        else r.setAttribute("aria-hidden", "true");
      });
      if (statusEl) {
        var h3 = readings[i].querySelector("h3");
        statusEl.textContent = "Reading " + (i + 1) + " of " + readings.length + ": " + (h3 ? h3.textContent : "");
      }
    }

    var endedHandler = null;
    function stopVideo(v) {
      if (!v) return;
      if (endedHandler) { v.removeEventListener("ended", endedHandler); endedHandler = null; }
      v.pause();
      v.removeAttribute("data-pending");
      syncLive(v);
    }

    /* pause is a set of reasons: leaving the tab, hovering to read, scrolling,
       tabbing in, or pressing the button. Playing means the set is empty. */
    var holds = {}, paused = false, frozen = false, wasPlaying = false, pauseAt = 0;

    function setHold(reason, on) {
      if (on) holds[reason] = 1; else delete holds[reason];
      var want = Object.keys(holds).length > 0;
      paused = want;
      /* the flag follows the timers, not the reasons: a sequence that starts
         while a hold is up must freeze at once, or it runs on with a dead bar */
      if (paused && !frozen) { frozen = true; freeze(); }
      else if (!paused && frozen) { frozen = false; thaw(); }
      toggleEl.setAttribute("aria-pressed", paused ? "true" : "false");
      toggleEl.setAttribute("aria-label", paused ? "Play the sequence" : "Pause the sequence");
    }

    function freeze() {
      pauseAt = Date.now();
      disarmTimers();
      cancelAnimationFrame(frame);
      var v = videos[current];
      wasPlaying = !!v && !v.paused;
      if (v) { v.pause(); syncLive(v); }
    }

    function thaw() {
      if (!playing || current < 0) return;
      var waited = Date.now() - pauseAt;
      stepStart += waited;
      armTimers();
      var v = videos[current];
      if (wasPlaying && v) setClip(v, true);
      if (current >= 0 && playing) {
        meter(current);
        if (steps[stepAt] && steps[stepAt].kind === "play") watchStep(current, steps[stepAt]);
      }
    }

    function stop() {
      clearTimers();
      cancelAnimationFrame(frame);
      frozen = false;
      playing = false;
      videos.forEach(stopVideo);
      readings.forEach(function (r) {
        r.classList.remove("is-on", "is-held");
        r.removeAttribute("data-beat");
        r.removeAttribute("aria-hidden");
      });
    }

    /* how far through the current step we are, 0 to 1 */
    function stepFraction() {
      var s = steps[stepAt];
      if (!s) return 1;
      if (s.kind === "play") {
        var v = videos[current];
        if (!v || !s.to) return 0;
        return clamp((v.currentTime - s.from) / (s.to - s.from), 0, 1);
      }
      if (!s.ms) return 1;
      return clamp((Date.now() - stepStart) / s.ms, 0, 1);
    }

    function meter(i) {
      segs.forEach(function (s, k) {
        s.classList.toggle("is-past", k < i);
        if (k !== i) s.style.setProperty("--p", k < i ? "1" : "0");
      });
      (function tick() {
        if (current !== i || !playing || paused) return;
        var done = 0;
        for (var k = 0; k < stepAt; k++) done += stepMs(steps[k]);
        var value = Math.min(1, (done + stepMs(steps[stepAt]) * stepFraction()) / total);
        segs[i].style.setProperty("--p", value.toFixed(3));
        /* the bar is the clock: full means the next reading starts, but only
           once — a step timer may already have handed over this frame */
        if (value >= 1) { if (current === i) advance(1); return; }
        frame = requestAnimationFrame(tick);
      })();
    }

    /* the four cycle; the last one hands back to the first */
    function advance(step) {
      if (!playing) return;
      var next = (current + step + N) % N;
      play(next);
    }

    /* an explicit step means "go", so the courtesies let go of the playhead */
    function resumeForAction() {
      delete holds.manual; delete holds.hover; delete holds.scroll; delete holds.focus;
      setHold("noop", false);
    }

    function render() {
      arrows.forEach(function (a) { a.disabled = false; });
    }

    function play(i) {
      stop();
      current = i;
      playing = true;
      var r = readings[i], v = videos[i];
      r.classList.add("is-on");
      focusTree(i);
      steps = stepsFor(i);
      stepAt = 0;
      total = steps.reduce(function (sum, s) { return sum + stepMs(s); }, 0) || 1;
      /* the first meter tick runs synchronously, so the clock has to be set
         before it, or it measures against the previous row's start */
      stepStart = Date.now();
      meter(i);
      runStep(i);
      if (paused && !frozen) { frozen = true; freeze(); }
      render();
    }

    function runStep(i) {
      var s = steps[stepAt], r = readings[i], v = videos[i];
      if (!s) { advance(1); return; }
      stepStart = Date.now();

      if (s.kind === "loop") {
        /* count whole passes rather than a stopwatch, so the clip always plays
           out instead of being cut mid-motion */
        s.passes = s.passes || 2;
        if (v.duration) {
          var need = readingMs(i);
          while (Math.round(v.duration * 1000 * s.passes) < need) s.passes++;
          s.ms = Math.round(v.duration * 1000 * s.passes);
          total = steps.reduce(function (sum, x) { return sum + stepMs(x); }, 0) || 1;
        }
        v.loop = false;
        try { v.currentTime = 0; } catch (e) {}
        setClip(v, true);
        later(function () { if (current === i) paint(r, 0); }, s.reveal * 1000);

        var played = 0;
        endedHandler = function () {
          if (current !== i || !playing) return;
          played++;
          if (played < s.passes) {
            try { v.currentTime = 0; } catch (e) {}
            setClip(v, true);
          } else {
            v.removeEventListener("ended", endedHandler);
            endedHandler = null;
            nextStep(i);
          }
        };
        v.addEventListener("ended", endedHandler);
        return;
      }

      if (s.kind === "hold") {
        if (v) { v.pause(); try { v.currentTime = s.at; } catch (e) {} syncLive(v); }
        paint(r, s.inset);
        later(function () { if (current === i) nextStep(i); }, s.ms);
        return;
      }

      /* the crop already on screen stays while the clip plays on: the dark
         end keeps its evidence until the bright one joins it */

      if (!v) { nextStep(i); return; }
      try { v.currentTime = s.from; } catch (e) {}
      setClip(v, true);
      (function watch() {
        if (current !== i || !playing) return;
        if (v.currentTime >= s.to - 0.03 || v.paused) { nextStep(i); return; }
        frame = requestAnimationFrame(watch);
      })();
    }

    function watchStep(i, s) {
      var v = videos[i];
      (function watch() {
        if (current !== i || !playing || paused) return;
        if (v.currentTime >= s.to - 0.03 || v.paused) { nextStep(i); return; }
        frame = requestAnimationFrame(watch);
      })();
    }

    function nextStep(i) {
      if (current !== i) return;
      stepAt++;
      if (stepAt >= steps.length) { advance(1); return; }
      runStep(i);
    }

    /* the crops size themselves against the plate, so measure it and keep the
       number current when the layout changes */
    readings.forEach(function (r) {
      /* the frame is the preview; the figure also holds a caption */
      var plate = r.querySelector(".plate__frame") || r.querySelector(".plate");
      if (!plate) return;
      var measure = function () {
        var frameH = plate.getBoundingClientRect().height;
        var caps = [].slice.call(r.querySelectorAll(".zoom figcaption"));
        /* outer height: the caption's own top margin sits between image and text */
        var capH = caps.reduce(function (sum, c) {
          return sum + c.getBoundingClientRect().height + parseFloat(getComputedStyle(c).marginTop || 0);
        }, 0);
        var n = Math.max(1, caps.length);
        var each = (frameH - capH - (n - 1) * 8 - 4) / n;
        r.style.setProperty("--plate-h", Math.round(frameH) + "px");
        r.style.setProperty("--crop-max", Math.max(40, Math.floor(each)) + "px");
      };
      measure();
      if ("ResizeObserver" in window) new ResizeObserver(measure).observe(plate);
    });

    videos.forEach(function (v, i) {
      if (!v) return;
      v.loop = plan[i].loop;
      v.preload = "auto";
      v.setAttribute("data-walk", "1");
    });

    /* the button is the explicit control; the rest are courtesies */
    toggleEl.addEventListener("click", function () {
      if (paused) { delete holds.hover; delete holds.focus; setHold("manual", false); }
      else setHold("manual", true);
    });

    /* a pointer resting on the media holds the sequence: that is where you
       look at the frame, and the reading should not change underneath you.
       Text and controls are left alone, and a finger is ignored. */
    var hoverIn = 0, hoverOut = 0;
    function mediaEnter(e) {
      if (e.pointerType !== "mouse") return;
      clearTimeout(hoverOut);
      clearTimeout(hoverIn);
      hoverIn = setTimeout(function () { setHold("hover", true); }, 240);
    }
    function mediaLeave(e) {
      if (e.pointerType !== "mouse") return;
      clearTimeout(hoverIn);
      /* a beat of grace: crossing from the plate to the crops is one gesture */
      clearTimeout(hoverOut);
      hoverOut = setTimeout(function () { setHold("hover", false); }, 180);
    }
    readings.forEach(function (r) {
      var media = r.querySelector(".reading__media");
      if (!media) return;
      media.addEventListener("pointerenter", mediaEnter);
      media.addEventListener("pointerleave", mediaLeave);
    });

    /* while the page is moving, the sequence waits; it catches up a beat later */
    var scrollIdle = 0;
    window.addEventListener("scroll", function () {
      if (!playing) return;
      setHold("scroll", true);
      clearTimeout(scrollIdle);
      scrollIdle = setTimeout(function () { setHold("scroll", false); }, 700);
    }, { passive: true });

    /* a keyboard user should not have the reading change underneath them */
    walkEl.addEventListener("focusin", function (e) {
      if (e.target && e.target.closest && e.target.closest(".walk__controls")) return;
      setHold("focus", true);
    });
    walkEl.addEventListener("focusout", function () {
      setTimeout(function () {
        if (!walkEl.contains(document.activeElement)) setHold("focus", false);
      }, 0);
    });

    document.addEventListener("visibilitychange", function () {
      setHold("hidden", document.hidden);
    });

    arrows.forEach(function (a) {
      a.addEventListener("click", function () {
        var step = parseInt(a.getAttribute("data-step"), 10);
        resumeForAction();
        play(current < 0 ? 0 : (current + step + N) % N);
      });
    });

    /* a sideways swipe steps the sequence, the way the arrows do */
    var t0x = 0, t0y = 0, tracked = false;
    walkEl.addEventListener("touchstart", function (e) {
      var t = e.changedTouches[0];
      t0x = t.clientX; t0y = t.clientY; tracked = true;
    }, { passive: true });
    walkEl.addEventListener("touchend", function (e) {
      if (!tracked) return;
      tracked = false;
      var t = e.changedTouches[0], dx = t.clientX - t0x, dy = t.clientY - t0y;
      if (Math.abs(dx) < 48 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
      resumeForAction();
      play((current + (dx < 0 ? 1 : -1) + N) % N);
    }, { passive: true });

    /* run while the section is on screen; stop the clock when it leaves */
    if ("IntersectionObserver" in window) {
      var seen = false;
      new IntersectionObserver(function (entries) {
        var on = entries[0].isIntersecting;
        if (on && !playing) {
          play(seen ? Math.max(0, current) : 0);
          seen = true;
        } else if (!on && playing) {
          var keep = Math.max(0, current);
          stop();
          current = keep;
          readings[current].classList.add("is-on");
        }
      }, { threshold: 0.35 }).observe(walkEl);
    } else {
      play(0);
    }

    render();
  }

  if (walkArmed) {
    walkEl.classList.add("is-armed");
    startWalk();
  }

  /* the event flood and the conventional view of the same window run together,
     so the flood lands on the frame where the lamp actually comes on */
  var eventsRow = walkArmed ? null : document.querySelectorAll(".reading")[2];
  var flood = eventsRow && eventsRow.querySelector(".clip video:not([data-follow])");
  var followers = walkArmed ? [] : [].slice.call(document.querySelectorAll(".reading .clip video[data-follow]"));

  [].forEach.call(walkArmed ? [] : document.querySelectorAll(".reading .clip video"), function (v) {
    new IntersectionObserver(function (entries) {
      var on = entries[0].isIntersecting;
      setClip(v, on);
      if (on && v !== flood) {
        try { v.currentTime = flood.currentTime; } catch (e) {}
      }
    }, { threshold: 0.3 }).observe(v);
  });

  if (flood && followers.length) {
    setInterval(function () {
      if (flood.paused) return;
      followers.forEach(function (v) {
        if (Math.abs(v.currentTime - flood.currentTime) > 0.12) {
          try { v.currentTime = flood.currentTime; } catch (e) {}
        }
      });
    }, 500);
  }

  var onNav = function () {
    if (nav) nav.classList.toggle("is-stuck", window.scrollY > 80);
  };
  window.addEventListener("scroll", onNav, { passive: true });
  onNav();
})();

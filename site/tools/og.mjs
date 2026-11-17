#!/usr/bin/env node
/*
  Regenerates site/og.png — the 1200x630 card that link previews show.

  The card is the real deck, captured from the page at its settled stop, so it
  tracks both the copy and the walk automatically. Run it after changing either.

    node site/tools/og.mjs

  Needs a Chromium. Either install puppeteer (which brings its own) or point
  CHROME_PATH at a binary you already have:

    CHROME_PATH=/usr/bin/chromium node site/tools/og.mjs
*/
import { writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = resolve(SITE, "og.png");

const CARD = {
  width: 1200,
  height: 630,
  headline: ["Perception is", "complementary."],
  standfirst: "One simulated scene, as seen by many sensors — with the exact ground truth behind every reading.",
  tagline: "one renderer · every sensor",
  stackWidth: 570,      // how wide the deck sits on the card
  stackMargin: 34,      // breathing room above and below it
  stackRight: 46,       // gap between the deck and the right edge
  /* front of the stack first. Thermal leads because the warm body reads at
     thumbnail size; events sits at the very back, where its white field becomes
     a bright rim against the page's near-black. */
  stackOrder: [3, 0, 2, 1, 4, 6, 5],
  /* the thermal plate leads, so it holds a chosen instant of its clip */
  thermalPlate: 3,
  thermalTime: 6.2,
};

const launch = async () => {
  try {
    const { default: puppeteer } = await import("puppeteer");
    return puppeteer.launch({ headless: "shell", args: ["--no-sandbox"] });
  } catch {
    const { default: puppeteer } = await import("puppeteer-core");
    let executablePath = process.env.CHROME_PATH;
    let args = ["--no-sandbox"];
    if (!executablePath) {
      /* environments that cannot download Chrome can supply one this way */
      const { default: chromium } = await import("@sparticuz/chromium");
      executablePath = await chromium.executablePath();
      args = [...chromium.args, "--no-sandbox"];
    }
    return puppeteer.launch({ headless: "shell", executablePath, args });
  }
};

const browser = await launch();
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 });
await page.goto(pathToFileURL(resolve(SITE, "index.html")).href, { waitUntil: "networkidle2" });

/* walk to the stop where the whole pile is out, and let the transition settle */
await page.evaluate(() => {
  const hero = document.querySelector(".hero");
  const top = window.scrollY + hero.getBoundingClientRect().top;
  window.scrollTo(0, top + (hero.getBoundingClientRect().height - innerHeight) * 0.93);
});
await new Promise((r) => setTimeout(r, 1500));

/* reorder the pile for the card: depth is written per plate by the page, so the
   order can be rewritten without touching the walk itself */
await page.evaluate((order) => {
  const plates = [...document.querySelectorAll(".deck__layer")];
  const deck = document.querySelector(".deck");
  const du = parseFloat(getComputedStyle(deck).getPropertyValue("--du")) || 60;
  order.forEach((plate, rank) => {
    plates[plate].style.setProperty("--zu", String(-rank * du));
    plates[plate].style.setProperty("--dim", "1");
    plates[plate].style.setProperty("--pr", "1");
  });
}, CARD.stackOrder);
/* hold the leading thermal plate on the moment worth showing */
await page.evaluate(async ({ plate, time }) => {
  const video = document.querySelectorAll(".deck__layer")[plate]?.querySelector("video");
  if (!video) return;
  video.preload = "auto";
  if (video.readyState < 2) await new Promise((r) => video.addEventListener("loadeddata", r, { once: true }));
  video.pause();
  video.currentTime = time;
  await new Promise((r) => video.addEventListener("seeked", r, { once: true }));
}, { plate: CARD.thermalPlate, time: CARD.thermalTime });

await new Promise((r) => setTimeout(r, 700));

const deck = await page.evaluate(() => {
  const plates = [...document.querySelectorAll(".deck__layer")].filter(
    (el) => +getComputedStyle(el).opacity > 0.15,
  );
  let l = Infinity, r = -Infinity, t = Infinity, b = -Infinity;
  for (const el of plates) {
    const box = el.getBoundingClientRect();
    l = Math.min(l, box.left); r = Math.max(r, box.right);
    t = Math.min(t, box.top); b = Math.max(b, box.bottom);
  }
  return { left: l, top: t, width: r - l, height: b - t, plates: plates.length, dpr: devicePixelRatio };
});
if (deck.plates < 7) console.warn(`only ${deck.plates} plates visible — the card may look sparse`);

/* hide everything but the deck, so the capture is plates on black and the card
   can position it without needing an image editor */
await page.evaluate(() => {
  for (const sel of [".hero__copy", ".rail", ".nav"]) {
    const el = document.querySelector(sel);
    if (el) el.style.visibility = "hidden";
  }
});

const shot = await page.screenshot({ encoding: "base64" });
await browser.close();

/* compose the card in the same browser, so the deck is cropped by CSS and the
   type is set in the site's own faces */
const html = `<!doctype html><meta charset="utf-8"><style>
@font-face { font-family: Hanken; src: url("${pathToFileURL(resolve(SITE, "fonts/hanken-grotesk-latin-400-normal.woff2")).href}") format("woff2"); font-weight: 400 }
@font-face { font-family: Hanken; src: url("${pathToFileURL(resolve(SITE, "fonts/hanken-grotesk-latin-700-normal.woff2")).href}") format("woff2"); font-weight: 700 }
@font-face { font-family: Spline; src: url("${pathToFileURL(resolve(SITE, "fonts/spline-sans-mono-latin-400-normal.woff2")).href}") format("woff2"); font-weight: 400 }
html, body { margin: 0; width: ${CARD.width}px; height: ${CARD.height}px; background: #0b0c0d; color: #f3f3f0;
  font-family: Hanken, sans-serif; overflow: hidden }
.card { position: relative; width: ${CARD.width}px; height: ${CARD.height}px; overflow: hidden }
.shot { position: absolute; transform-origin: 0 0 }
.copy { position: absolute; left: 64px; top: 128px; width: 470px }
h1 { margin: 0; font-size: 58px; line-height: 1.03; letter-spacing: -.036em; font-weight: 400 }
p { margin: 24px 0 0; font-size: 20px; line-height: 1.5; color: rgba(243,243,240,.66); max-width: 28ch }
.tag { position: absolute; left: 64px; top: 56px; font-family: Spline, monospace; font-size: 12px;
  letter-spacing: .18em; text-transform: uppercase; color: #767b79 }
.brand { position: absolute; left: 64px; bottom: 52px; display: flex; align-items: center; gap: 12px;
  font-weight: 700; font-size: 25px; letter-spacing: -.015em }
.brand span { display: inline-block }
.brand em { font-style: normal; color: #a6a9a7; font-weight: 400 }
</style>
<div class="card">
  <img class="shot" src="data:image/png;base64,${shot}">
  <div class="tag">${CARD.tagline}</div>
  <div class="copy">
    <h1>${CARD.headline.join("<br>")}</h1>
    <p>${CARD.standfirst}</p>
  </div>
  <div class="brand">
    <svg width="32" height="32" viewBox="0 0 26 26"><g fill="none" stroke="#c02d0c" stroke-width="2.2"><rect x="1.9" y="4.3" width="16.4" height="11.6" rx="2.5" opacity=".42"/><rect x="5.3" y="7.7" width="16.4" height="11.6" rx="2.5" opacity=".74"/></g><rect x="8.7" y="11.1" width="16.4" height="11.6" rx="2.5" fill="#c02d0c"/></svg>
    <span>Vision<em>SIM</em></span>
  </div>
</div>`;

const renderer = await (await launch()).newPage();
/* the capture is at dpr 2, so one page CSS px is dpr image px; fit both ways
   so the plates and their shadows stay inside the card */
const scale = Math.min(
  CARD.stackWidth / (deck.width * deck.dpr),
  (CARD.height - CARD.stackMargin * 2) / (deck.height * deck.dpr),
);
const x = CARD.width - CARD.stackRight - CARD.stackWidth;
const y = Math.round((CARD.height - deck.height * deck.dpr * scale) / 2);

await renderer.setViewport({ width: CARD.width, height: CARD.height, deviceScaleFactor: 1 });
await renderer.setContent(html, { waitUntil: "load" });
await renderer.evaluate(
  ({ scale, x, y, left, top, dpr }) => {
    const img = document.querySelector(".shot");
    img.style.width = `${img.naturalWidth}px`;   /* natural size = viewport * dpr */
    img.style.transform = `translate(${x - left * dpr * scale}px, ${y - top * dpr * scale}px) scale(${scale})`;
  },
  { scale, x, y, left: deck.left, top: deck.top, dpr: deck.dpr },
);
await renderer.evaluate(() => document.fonts.ready);
await new Promise((r) => setTimeout(r, 250));

const card = await renderer.screenshot({ type: "png" });
await writeFile(OUT, card);
await renderer.close();

console.log(`wrote ${OUT} (${CARD.width}x${CARD.height}) from ${deck.plates} plates`);
process.exit(0);

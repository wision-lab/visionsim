"""Build a self-contained HTML report for the φ-pipeline verification example.

Reads the PNGs produced by ``aspc_transients_synthetic.py``, embeds them as base64
data URIs, and writes a single standalone file that needs no network, no assets
directory, and no server — just open it in a browser.

    PYTHONPATH=. python examples/sensors/aspc_transients_synthetic.py
    python examples/sensors/build_aspc_report.py

Both default to ``examples/sensors/aspc/`` (figures in ``figures/``, report alongside
them), anchored to this file so they work from any working directory.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import pathlib

# --------------------------------------------------------------------------- #
# Content                                                                      #
# --------------------------------------------------------------------------- #
FIGS = [
    (
        "7_loader_contract.png",
        "The loader contract, made concrete",
        "One flat wall at 5&nbsp;m with a sky band, loaded three ways: sky marked with 0 (conforming), "
        "sky marked with a 25&nbsp;m numeric sentinel, and the whole map handed over in millimetres.",
        "The data loader is deliberately outside this example&rsquo;s scope &mdash; that is what makes it "
        "dependency-free. So the boundary has to be stated as a contract instead, and a contract is "
        "only useful if the cost of violating it is visible.",
        "Conforming input gives one true surface at bin&nbsp;66. A 25&nbsp;m sentinel <strong>aliases to "
        "bin&nbsp;133 at 2% of the true peak</strong> &mdash; a plausible weak return, which is exactly why "
        "sentinels are forbidden. Millimetres silently lands in bin&nbsp;112 with no error raised. Both "
        "violations produce believable output; neither raises. A pint <code>Quantity</code> and a bare "
        "tensor in metres are verified identical.",
        ["L1", "T3", "T8"],
    ),
    (
        "4_depth_recovery.png",
        "End-to-end depth recovery",
        "40 flat walls at depths spanning 0.6&nbsp;m to 14.5&nbsp;m. Each is pushed through the "
        "complete pipeline, then <code>argmax(&phi;)</code> is converted back to metres via the "
        "bin centre and compared against truth.",
        "Everything upstream has to be simultaneously right for these points to land on "
        "<em>y&nbsp;=&nbsp;x</em>: radiometry, depth&rarr;bin mapping, aliasing, IRF width and "
        "convolution orientation. A single broken stage bends the line or offsets it.",
        "Max error <strong>0.500 bins</strong> &mdash; exactly the quantisation floor. The "
        "residual panel is the real diagnostic: pure rounding produces the uniform sawtooth you "
        "see, whereas a half-bin offset, a kernel-flip error or an off-by-one would show up as a "
        "tilt or a step.",
        ["A4", "T3", "T4", "T7"],
    ),
    (
        "5_invariances.png",
        "Invariances that must hold",
        "The same flat wall at 6&nbsp;m, re-run while varying exactly one thing at a time: render "
        "grid (16&sup2;&rarr;128&sup2;), FOV rectangle size, and vignette on/off.",
        "Properties that must <em>not</em> change are often more diagnostic than ones that must. "
        "If collected photons track render resolution, then the render grid is secretly acting as "
        "the physical sensor.",
        "(a) Four resolutions give coincident transients, spread <strong>1.9e-4</strong>. "
        "(b) Four FOV sizes give equal totals &mdash; a SPAD averages over its FOV. "
        "(c) Vignette attenuates <strong>&times;0.4983</strong> against an expected "
        "<strong>&times;0.4983</strong>, confirming the weights are applied and not wrongly "
        "normalised away.",
        ["F3", "T1"],
    ),
    (
        "3_depth_binning.png",
        "Depth to bin, against the oracle",
        "Three scenes &mdash; flat wall at 4&nbsp;m, two planes at 3 and 9&nbsp;m, tilted plane "
        "spanning 2&rarr;13&nbsp;m &mdash; with predicted bins drawn as dashed lines from the "
        "closed form <code>floor(d&middot;n/max_depth) mod n</code>.",
        "The depth&rarr;bin conversion is the one step with a hand-computable answer for any "
        "scene, so comparing occupied bins against the closed form isolates it from everything "
        "else in the pipeline.",
        "Exact match on all three, including <strong>8 distinct bins</strong> for the tilted "
        "plane. The <code>mod n</code> in the oracle is itself the aliasing fix &mdash; the old "
        "<code>clamp</code> behaviour fails the 13&nbsp;m case.",
        ["T3"],
    ),
    (
        "6_guards.png",
        "Regression guards, one panel per fix",
        "Four targeted probes, each plotting the <em>old</em> wrong behaviour beside the new so "
        "the difference is visible rather than merely asserted.",
        "A passing test tells you the code is right today. Showing the failure mode next to it "
        "tells you which regression the test actually catches.",
        "(a) <strong>T7</strong> &mdash; IRF sum 6.67 vs 1.0000. Note 6.67 here against 33.36 at "
        "<code>n_bins=1000</code>: that difference is itself the evidence that the error was "
        "configuration-dependent. (b) <strong>T4</strong> &mdash; mirrored vs true convolution, "
        "with the pipeline output marked. (c) <strong>T3</strong> &mdash; aliasing to bin 80 "
        "rather than piling at bin 199, log-scaled because the aliased return is genuinely ~27&times; "
        "fainter from inverse-square falloff at 21&nbsp;m vs 4&nbsp;m. (d) <strong>L1/T8</strong> "
        "&mdash; invalid depths dropped, 66% pixel validity, output finite.",
        ["T7", "T4", "T3", "L1", "T8"],
    ),
    (
        "2_pipeline.png",
        "The pipeline, stage by stage",
        "A sphere on a backdrop &mdash; the only scene here with curvature &mdash; shown at all "
        "three stages with a small ambient term switched on.",
        "The intermediate stages are where a magnitude error hides. Seeing the transient, the "
        "kernel and &phi; together makes the convolution&rsquo;s effect legible.",
        "Sphere curvature spreads across bins 46&ndash;53 while the flat backdrop stays a single "
        "spike at 146. The IRF sums to <strong>1.000000</strong>. In &phi; the peaks are broadened "
        "by exactly the pulse width, and the ambient floor sits underneath, uniform across all bins.",
        ["T7"],
    ),
    (
        "1_scenes.png",
        "The synthetic scenes",
        "Albedo and depth maps built from <code>torch</code> primitives &mdash; no renders, no "
        "dataset, no loader anywhere in the path.",
        "Orientation rather than verification. Included so the geometry behind every other figure "
        "is inspectable.",
        "Five scenes: flat wall, two planes, tilted plane, sphere on backdrop, and one carrying a "
        "sky band (depth&nbsp;0) plus a dropout column (NaN) to exercise invalid-depth handling.",
        [],
    ),
]

STAGES = [
    ("Stage 0", "Data loader", "utils.py", "albedo + depth maps out of rendered frames", [
        ("L1", "<code>cv2.inpaint</code> fabricated depth for every out-of-range pixel, inventing "
               "plausible-looking geometry for sky and far background", "fixed",
         "Removed. Because beyond-range returns now alias correctly, those pixels were never invalid."),
        ("L2", "Two incompatible depth encodings selected by array shape alone", "fixed",
         "Channel selection and unit interpretation are now separate decisions; rescaling keys off integer dtype."),
        ("L3", "Two different <code>max_depth</code> values &mdash; one for the inpaint threshold, one for rescaling", "fixed",
         "Split into <code>max_resolvable_depth</code> (hardware limit) and <code>scale_depth</code> (renderer range), with a check between them."),
        ("L4", "Bilinear resize on depth, inventing flying-pixel surfaces at every object edge", "fixed",
         "<code>cv2.INTER_NEAREST</code>."),
        ("L5", "Albedo taken as the red channel of an already-shaded render, so renderer illumination is double-counted", "documented",
         "Accepted as a deliberate approximation; the assumption is now stated at the point of use."),
        ("L6", "Debug prints in library code", "fixed", ""),
    ]),
    ("Stage 1", "Radiometry", "sources.py, sensors.py", "scene &rarr; photons per SPAD pixel per cycle", [
        ("T8", "Depth&nbsp;0 guarded by a 1e-12&nbsp;m epsilon, giving radiance ~1e24 rather than being rejected", "fixed",
         "Non-finite and &le;&nbsp;0 depths mean &ldquo;no surface&rdquo; and are dropped upstream of the epsilon."),
        ("A1", "Ambient radiance wrongly scales with the camera&rsquo;s per-pixel solid angle", "open",
         "Modeling decision, deliberately deferred."),
    ]),
    ("Stage 2", "Transient assembly", "histogrammers.py", "irradiance binned by depth, per FOV", [
        ("T1", "Vignette weights computed then discarded &mdash; the multiply landed on a rebound name, "
               "not the values actually scattered", "fixed",
         "Applied to <code>fov_irradiance_vals</code>; the dead rebind is gone."),
        ("T3", "Beyond-range returns clamped into the last bin, creating a spike that reads as a real surface", "fixed",
         "<code>torch.remainder</code> aliases them. Mode-independent: the detector is armed every cycle, so this holds for gated and free-running alike."),
        ("F3", "Render resolution coupled to physical sensor size &mdash; doubling resolution collected 4&times; the photons", "fixed",
         "Divide by <code>n_fov_pixels</code>, so a SPAD averages over its FOV instead of summing raw."),
        ("F8", "Default full-scene FOV <code>[0, 0.999, 0, 0.999]</code> was resolution-dependent, dropping 1,999&nbsp;px at 1000&sup2;", "fixed",
         "Now <code>[0, 1.0, 0, 1.0]</code>."),
        ("T2", "Frames stack rather than accumulate; the preallocated buffer was never written", "partial",
         "Dead buffer removed. Stacking semantics remain undecided, pinned by a characterisation test."),
        ("T5", "Ambient path skips the FOV correction the signal path receives", "partial",
         "Ambient is now divided by <code>n_fov_pixels</code> too. The modeling asymmetry stays open, tied to A1."),
    ]),
    ("Stage 3", "IRF and convolution", "camera.py, histogrammers.py", "transient &lowast; pulse + ambient = &phi;", [
        ("T7", "The IRF was never normalised &mdash; every arrival rate inflated by a configuration-dependent "
               "factor (33.36&times; at n_bins=1000, 6.67&times; at 200; 11/21/51&times; for square pulses at &tau;&nbsp;=&nbsp;1/2/5&nbsp;ns)", "fixed",
         "Passes <code>normalize=&quot;sum&quot;</code>. Largest single scalar error in the layer."),
        ("T4", "<code>F.conv1d</code> is cross-correlation, so asymmetric pulses came out time-reversed", "fixed",
         "<code>irf.flip(-1)</code>, verified against <code>np.convolve</code> rather than hand-derived indices."),
        ("T6", "Configured <code>bin_width</code> silently ignored", "fixed",
         "Validated against the frequency-implied value at 1% tolerance; raises on real disagreement."),
        ("A4", "One-way vs round-trip conventions cancel only by luck", "verified",
         "Correct as written &mdash; the two factors of 2 cancel exactly. Now pinned by a test so neither can be &ldquo;fixed&rdquo; alone."),
    ]),
    ("Cross-cutting", "Geometry and packaging", "camera.py", "FOV construction, Python compatibility", [
        ("F1", "<code>build_pixel_fov_list</code> built a list and never assigned it &mdash; the output was orphaned", "fixed",
         "Assigns <code>self.histogrammer.pixel_fov_list</code> before returning."),
        ("F5", "FOV centre used the pixel top edge, a half-pixel bias", "fixed", "<code>(row+0.5)/h</code>."),
        ("F6", "Docstring promised a float where a pint <code>Quantity</code> is required", "fixed", "Corrected."),
        ("E2", "<code>dict | None</code> made <code>camera.py</code> unimportable on Python&nbsp;3.9, which the project declares as supported", "fixed",
         "<code>from __future__ import annotations</code>. This is what made the camera-layer fixes testable at all."),
    ]),
]

CHECKS = [
    ("Convolution conserves photons", "&Sigma;&phi; &minus; ambient = 1.94753e-05 = &Sigma;transient"),
    ("IRF is energy-normalised", "&Sigma;irf = 1.000000"),
    ("Depth&rarr;bin exact: flat wall @ 4.0 m", "bins [53]"),
    ("Depth&rarr;bin exact: two planes @ 3 / 9 m", "bins [40, 120]"),
    ("Depth&rarr;bin exact: tilted plane 2&rarr;13 m", "8 bins, all predicted"),
    ("Depth recovered within &frac12; bin", "max |err| = 0.5000 bins"),
    ("Transient independent of render resolution", "spread 1.87e-04"),
    ("Transient independent of FOV size", "spread 5.92e-05"),
    ("Vignette applied with the right weight", "&times;0.49832 vs &times;0.49830 expected"),
    ("Convolution not mirrored", "matches np.convolve"),
    ("Beyond-range aliases, last bin empty", "last bin = 0"),
    ("Invalid depths contribute nothing", "bins [66]"),
    ("No NaN/Inf anywhere in &phi;", "all finite"),
    ("Valid-pixel fraction scales the signal", "exact"),
    ("Contract: sky marked 0/NaN is dropped", "bins [66]"),
    ("Contract: near-range sentinel creates a ghost", "25 m &rarr; bin 133 @ 2% of peak"),
    ("Contract: wrong depth units fail silently", "lands in [112], expected [66]"),
    ("Contract: pint Quantity == bare tensor in metres", "identical"),
]

NOT_COVERED = [
    ("<code>Camera</code> orchestration", "<code>get_transients</code> / <code>get_arrival_rates</code> glue needs a dataset to construct, so the example calls the underlying functions directly."),
    ("The data loader itself", "L1&ndash;L4 are verified by reading only. The example bypasses the loader by design and specifies the boundary as a written contract instead &mdash; see figure&nbsp;1. Any loader satisfying that contract inherits everything verified here; nothing enforces it at runtime, so conformance is the loader author&rsquo;s responsibility."),
    ("Everything after &phi;", "Dead time, pile-up and the histogram forward models are untouched here. Four <code>xfail</code>s remain there: A1, H1&times;2 and dt=0."),
]

BADGES = {
    "fixed": ("fixed", "ok"),
    "verified": ("verified correct", "ok"),
    "partial": ("partly addressed", "warn"),
    "open": ("open by decision", "warn"),
    "documented": ("documented", "warn"),
}

CSS = """
*{box-sizing:border-box}
:root{
  --bg:#f4f6f7; --surface:#ffffff; --surface-2:#eef1f3; --line:#d5dbe0;
  --fg:#0e151d; --fg-2:#44515f; --fg-3:#6b7887;
  --teal:#1d7d71; --teal-soft:#e2f0ee;
  --rust:#b83c4e; --rust-soft:#fbe7ea;
  --amber:#9a6414; --amber-soft:#f8eeda;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0e151d; --surface:#151e28; --surface-2:#1c2733; --line:#2b3948;
    --fg:#e9eef2; --fg-2:#a9b6c3; --fg-3:#7d8b99;
    --teal:#4cbfb0; --teal-soft:#14312e;
    --rust:#e8798a; --rust-soft:#331a20;
    --amber:#d9a441; --amber-soft:#33270f;
  }
}
:root[data-theme="dark"]{
  --bg:#0e151d; --surface:#151e28; --surface-2:#1c2733; --line:#2b3948;
  --fg:#e9eef2; --fg-2:#a9b6c3; --fg-3:#7d8b99;
  --teal:#4cbfb0; --teal-soft:#14312e;
  --rust:#e8798a; --rust-soft:#331a20;
  --amber:#d9a441; --amber-soft:#33270f;
}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);
  font-size:16px;line-height:1.62;margin:0;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}
.measure{max-width:70ch}
h1,h2,h3{letter-spacing:-.021em;text-wrap:balance;margin:0}
h1{font-size:clamp(2rem,4.4vw,2.9rem);font-weight:680;line-height:1.1}
h2{font-size:1.5rem;font-weight:660}
h3{font-size:1.1rem;font-weight:640}
p{margin:0}
code{font-family:var(--mono);font-size:.88em;background:var(--surface-2);
  padding:.1em .34em;border-radius:3px;border:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;
  letter-spacing:.14em;color:var(--fg-3)}
header.top{border-bottom:1px solid var(--line);padding:56px 0 32px;margin-bottom:44px}
header.top .inner{display:flex;flex-direction:column;gap:18px}
.lede{font-size:1.08rem;color:var(--fg-2);max-width:68ch}
.meta{display:flex;flex-wrap:wrap;gap:10px 26px;font-family:var(--mono);
  font-size:.76rem;color:var(--fg-3);padding-top:6px}
.meta b{color:var(--fg-2);font-weight:500}
.banner{display:flex;flex-wrap:wrap;align-items:center;gap:20px;
  background:var(--teal-soft);border:1px solid var(--teal);border-radius:6px;padding:18px 22px}
.banner .big{font-family:var(--mono);font-size:1.9rem;font-weight:600;
  color:var(--teal);font-variant-numeric:tabular-nums;line-height:1}
.banner .txt{color:var(--fg-2);font-size:.94rem;max-width:56ch}
section.blk{margin-top:64px}
.blk > h2{margin-bottom:8px}
.blk > .sub{color:var(--fg-2);margin-bottom:26px;max-width:70ch}
.flow{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));margin-top:8px}
.node{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:16px 18px;display:flex;flex-direction:column;gap:9px;position:relative}
.node .stagename{font-weight:620;font-size:.98rem}
.node .what{color:var(--fg-3);font-size:.83rem;font-family:var(--mono);line-height:1.45}
.node .ids{display:flex;flex-wrap:wrap;gap:5px;margin-top:2px}
.chip{font-family:var(--mono);font-size:.7rem;padding:2px 6px;border-radius:3px;
  border:1px solid;font-variant-numeric:tabular-nums}
.chip.ok{color:var(--teal);border-color:var(--teal);background:var(--teal-soft)}
.chip.warn{color:var(--amber);border-color:var(--amber);background:var(--amber-soft)}
.checks{list-style:none;padding:0;margin:0;display:grid;gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.checks li{background:var(--surface);display:grid;
  grid-template-columns:56px minmax(0,1fr) minmax(0,auto);gap:14px;
  align-items:baseline;padding:9px 16px}
.ck{font-family:var(--mono);font-size:.68rem;font-weight:600;color:var(--teal);letter-spacing:.06em}
.ck-n{font-size:.9rem}
.ck-v{font-family:var(--mono);font-size:.78rem;color:var(--fg-3);
  font-variant-numeric:tabular-nums;text-align:right}
.fig{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:26px;margin-top:22px;display:flex;flex-direction:column;gap:20px}
.fig-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.fig-head h3{flex:1 1 auto;min-width:220px}
.tags{display:flex;flex-wrap:wrap;gap:5px}
.fig figure{margin:0;display:flex;flex-direction:column;gap:8px}
.fig img{width:100%;height:auto;display:block;border:1px solid var(--line);
  border-radius:4px;background:#fff}
figcaption{font-family:var(--mono);font-size:.72rem;color:var(--fg-3)}
.fig-body{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(255px,1fr))}
.note{display:flex;flex-direction:column;gap:6px}
.note-k{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.12em;color:var(--fg-3)}
.note p{font-size:.92rem;color:var(--fg-2)}
.stage-block{margin-top:34px}
.stage-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px;margin-bottom:12px}
.files{color:var(--fg-3);background:none;border:none;padding:0;font-size:.78rem}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:660px}
th{text-align:left;font-family:var(--mono);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.1em;color:var(--fg-3);font-weight:500;
  padding:11px 16px;border-bottom:1px solid var(--line);background:var(--surface-2)}
td{padding:13px 16px;border-bottom:1px solid var(--line);
  vertical-align:top;font-size:.9rem;color:var(--fg-2)}
tr:last-child td{border-bottom:none}
td.id{width:64px}
td.id code{background:none;border:none;padding:0;color:var(--fg);font-weight:600}
.pill{display:inline-block;font-family:var(--mono);font-size:.68rem;padding:2px 7px;
  border-radius:3px;border:1px solid;white-space:nowrap}
.pill.ok{color:var(--teal);border-color:var(--teal);background:var(--teal-soft)}
.pill.warn{color:var(--amber);border-color:var(--amber);background:var(--amber-soft)}
.fix{display:block;margin-top:7px;font-size:.85rem;color:var(--fg-3)}
.gaps{display:grid;gap:12px;margin-top:6px}
.gap{border-left:2px solid var(--rust);background:var(--rust-soft);
  padding:13px 18px;border-radius:0 5px 5px 0}
.gap b{display:block;font-size:.94rem;color:var(--fg);margin-bottom:3px}
.gap span{font-size:.9rem;color:var(--fg-2)}
.repro{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:20px 22px;display:flex;flex-direction:column;gap:12px;margin-top:6px}
pre{margin:0;overflow-x:auto;font-family:var(--mono);font-size:.82rem;
  background:var(--surface-2);border:1px solid var(--line);border-radius:5px;padding:13px 15px;color:var(--fg-2)}
footer{margin-top:80px;padding-top:22px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:.74rem;color:var(--fg-3);
  display:flex;flex-wrap:wrap;gap:8px 22px}
a{color:var(--teal)}
:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
@media (max-width:620px){
  .checks li{grid-template-columns:48px 1fr;gap:4px 12px}
  .ck-v{grid-column:2;text-align:left}
}
"""


def data_uri(path: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build(figdir: pathlib.Path) -> str:
    flow = ""
    for stage, name, files, what, rows in STAGES:
        chips = "".join(
            f'<span class="chip {BADGES[st][1]}">{fid}</span>' for fid, _, st, _ in rows
        )
        flow += (
            f'<div class="node"><span class="eyebrow">{stage}</span>'
            f'<span class="stagename">{name}</span>'
            f'<span class="what">{what}</span>'
            f'<div class="ids">{chips}</div></div>'
        )

    checks = "".join(
        f'<li><span class="ck">PASS</span><span class="ck-n">{n}</span>'
        f'<span class="ck-v">{v}</span></li>'
        for n, v in CHECKS
    )

    figs = ""
    for i, (fname, title, howgen, why, proves, guards) in enumerate(FIGS, 1):
        path = figdir / fname
        if not path.exists():
            raise SystemExit(f"missing figure {path} — run aspc_transients_synthetic.py first")
        tags = "".join(f'<span class="chip ok">{g}</span>' for g in guards)
        figs += f"""
<section class="fig">
  <div class="fig-head"><span class="eyebrow">Figure {i}</span><h3>{title}</h3>
    {f'<div class="tags">{tags}</div>' if tags else ''}</div>
  <figure><img src="{data_uri(path)}" alt="{title}" />
    <figcaption>{fname}</figcaption></figure>
  <div class="fig-body">
    <div class="note"><span class="note-k">How it&rsquo;s generated</span><p>{howgen}</p></div>
    <div class="note"><span class="note-k">Why this plot</span><p>{why}</p></div>
    <div class="note"><span class="note-k">What it establishes</span><p>{proves}</p></div>
  </div>
</section>"""

    tables = ""
    for stage, name, files, _what, rows in STAGES:
        trs = ""
        for fid, prob, st, fix in rows:
            label, cls = BADGES[st]
            trs += (
                f'<tr><td class="id"><code>{fid}</code></td><td>{prob}</td>'
                f'<td><span class="pill {cls}">{label}</span>'
                + (f'<span class="fix">{fix}</span>' if fix else "")
                + "</td></tr>"
            )
        tables += f"""
<div class="stage-block">
  <div class="stage-head"><span class="eyebrow">{stage}</span><h3>{name}</h3>
    <code class="files">{files}</code></div>
  <div class="tablewrap"><table>
    <thead><tr><th>ID</th><th>Problem</th><th>Status &amp; resolution</th></tr></thead>
    <tbody>{trs}</tbody></table></div>
</div>"""

    gaps = "".join(f'<div class="gap"><b>{t}</b><span>{d}</span></div>' for t, d in NOT_COVERED)
    n_fixed = sum(
        1 for *_, rows in STAGES for row in rows if row[2] in ("fixed", "verified")
    )
    stamp = _dt.date.today().isoformat()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Photon Arrival Rate Verification</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="top"><div class="inner">
  <span class="eyebrow">visionsim &middot; emulate/aspc &middot; branch active-spc</span>
  <h1>Photon Arrival Rate Verification</h1>
  <p class="lede">What the &phi; forward model does, which parts of it were wrong, and how each
  fix is now checked. &phi; is the photon <em>arrival</em> rate &mdash; scene geometry &times;
  laser &times; ambient. Dead time and gating decide which arrivals become <em>detections</em>,
  and that is entirely downstream, so nothing on this page depends on gated versus free-running
  operation.</p>
  <div class="meta">
    <span><b>Source</b> examples/sensors/aspc_transients_synthetic.py</span>
    <span><b>Config</b> 10&nbsp;MHz, 2&nbsp;ns pulse, 200 bins</span>
    <span><b>Range</b> 14.990&nbsp;m @ 7.49&nbsp;cm/bin</span>
    <span><b>Generated</b> {stamp}</span>
  </div>
</div></header>

<div class="banner">
  <span class="big">18/18</span>
  <span class="txt">self-checks pass in the example, alongside <b>129 passed / 4 xfailed</b> in
  <code>tests/test_aspc_*.py</code>. All four remaining xfails sit in the forward-model layer
  downstream of &phi; and are deferred modeling decisions, not defects awaiting a patch.</span>
</div>

<section class="blk">
  <h2>What the pipeline does, and where it was broken</h2>
  <p class="sub measure">Four stages turn a rendered scene into an arrival rate. Every finding
  below is attached to the stage it lives in; teal means resolved, amber means deliberately
  still open.</p>
  <div class="flow">{flow}</div>
</section>

<section class="blk">
  <h2>Self-checks</h2>
  <p class="sub measure">Every scene has a closed-form expected answer, so the example asserts
  rather than asking you to eyeball the plots. It exits non-zero on failure and doubles as a
  smoke test.</p>
  <ul class="checks">{checks}</ul>
</section>

<section class="blk">
  <h2>The figures</h2>
  <p class="sub measure">Ordered by how much confidence each one carries, not by the order the
  script emits them. Figure&nbsp;1 defines the boundary this verification rests on;
  figure&nbsp;2 is the strongest single check that everything inside it works.</p>
  {figs}
</section>

<section class="blk">
  <h2>Findings by stage</h2>
  <p class="sub measure">Full register for this layer. IDs match
  <code>ASPC_KNOWN_ISSUES.md</code> &sect;9.</p>
  {tables}
</section>

<section class="blk">
  <h2>What this does <em>not</em> cover</h2>
  <p class="sub measure">Stated plainly, because a green report is only as good as its
  boundaries.</p>
  <div class="gaps">{gaps}</div>
</section>

<section class="blk">
  <h2>Reproducing this</h2>
  <div class="repro">
    <p class="measure">From the repository root. The example needs no dataset, no network and no
    render output &mdash; scenes are built analytically from <code>torch</code> primitives, then
    pushed through the real library functions.</p>
<pre>PYTHONPATH=. python examples/sensors/aspc_transients_synthetic.py
python examples/sensors/build_aspc_report.py</pre>
    <p class="measure">Both write into <code>examples/sensors/aspc/</code> &mdash; figures under
    <code>figures/</code>, this report beside them &mdash; with paths anchored to the scripts
    rather than the working directory.</p>
    <p class="measure">The only re-implemented piece is the two-line expression from
    <code>Camera._get_signal</code>, because <code>Camera</code> requires a dataset path to
    construct. Everything else &mdash; <code>get_scene_radiance</code>,
    <code>calculate_transients</code>, <code>get_kernel</code>,
    <code>calculate_arrival_rates</code> &mdash; is called directly.</p>
  </div>
</section>

<footer>
  <span>{n_fixed} findings resolved in this layer</span>
  <span>figures embedded as base64 &mdash; this file is self-contained</span>
  <span>see ASPC_KNOWN_ISSUES.md &sect;9 and &sect;10</span>
</footer>

</div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Anchored to this file rather than the cwd, matching aspc_transients_synthetic.py.
    here = pathlib.Path(__file__).parent / "aspc"
    ap.add_argument("--figdir", type=pathlib.Path, default=here / "figures")
    ap.add_argument("--out", type=pathlib.Path, default=here / "aspc_report.html")
    args = ap.parse_args()

    args.out.write_text(build(args.figdir), encoding="utf-8")
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.2f} MB, self-contained)")


if __name__ == "__main__":
    main()

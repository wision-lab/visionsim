"""Build a self-contained HTML report for the ground-truth detection model.

Companion to ``build_aspc_report.py``, which covers the φ pipeline. This one covers
what happens after φ: which arrivals become detections, under each operating mode
and dead-time model.

The report is written to answer one question — *is the implementation correct, or
is something wrong?* — so every figure carries an explicit *what a broken
implementation would look like here*, and the evidence is graded by how
independent it is rather than presented as a flat list of green ticks.

    PYTHONPATH=. python examples/sensors/aspc_detection_groundtruth.py
    python examples/sensors/build_aspc_detection_report.py

Check results are read from ``detection_checks.tsv``, written by the example, so
the report always quotes the run that produced the figures beside it.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_aspc_report import CSS  # noqa: E402  reuse the sibling report's styling

# --------------------------------------------------------------------------- #
# Verdict                                                                      #
# --------------------------------------------------------------------------- #
VERDICT = (
    "No implementation error found.",
    "Every prediction that can be checked against a source independent of the "
    "simulator agrees with it: detection rates match closed-form renewal theory to "
    "3.7%, the Coates estimator inverts simulated pile-up back to the true &phi;, and "
    "an exhaustive enumeration over every possible arrival pattern reproduces an "
    "independently written reference exactly. Two previously broken behaviours "
    "(<code>M2</code>, <code>M3</code>) are fixed, and a third open finding "
    "(<code>dt=0</code>) turns out to have been blaming the wrong component.",
)

# Grading the evidence matters more than counting it: most of these figures could
# be made to pass by a simulator that is confidently and consistently wrong.
TIERS = [
    (
        "strong", "Independent oracle",
        "Checked against a result derived separately from the implementation, so a "
        "shared misunderstanding cannot make both agree.",
        [
            "Non-paralyzable detection rate vs renewal-theory closed form &mdash; 14/14 flux levels, max deviation 3.7%",
            "Paralyzable detection rate vs its closed form &mdash; 11/14 well-sampled levels, max deviation 3.7%",
            "Coates estimator inverts simulated pile-up back to flat &phi; &mdash; worst bin 2.1&sigma; of counting noise",
            "Inter-detection gap tail recovers the arrival rate &mdash; true &lambda;=0.03, recovered 0.0300 / 0.0303 / 0.0299",
            "Exhaustive enumeration of all 3<sup>8</sup> arrival patterns vs a per-photon reference &mdash; exact match, no sampling error",
        ],
    ),
    (
        "medium", "Hard invariant",
        "Properties that must hold on any valid output, and that a large class of "
        "dead-time bugs would violate immediately.",
        [
            "No two detections ever closer than the dead time &mdash; holds at every dt tested, including dt &gt; one cycle",
            "Gated single-hit never exceeds one detection per cycle",
            "Free-running detects strictly fewer photons than gated multi-hit on identical arrivals (2.77% fewer)",
            "Paralyzable never detects more than non-paralyzable, and the two converge at low flux",
            "Detection rate decreases monotonically in dead time and saturates at 1/&tau;",
        ],
    ),
    (
        "weak", "Qualitative signature",
        "Emergent behaviour that is right in shape. Reassuring, but would not catch "
        "an error in scale.",
        [
            "Paralyzable rate rises then collapses; non-paralyzable saturates &mdash; the two device types differ in kind",
            "Gating biases the measured return earlier in the cycle (centre of mass 5.70&nbsp;m vs true 8.05&nbsp;m)",
            "Raw gated histograms show exponential pile-up decay under a deliberately flat &phi;",
        ],
    ),
]

# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
FIGS = [
    (
        "12_rate_vs_flux.png",
        "Detection rate vs flux, against closed-form oracles",
        "Flat &phi; swept over four decades at a fixed dead time of 10 bins, run through both "
        "dead-time models, and plotted against rates derived independently from renewal theory: "
        "<code>1/(&tau; + (1&minus;p)/p)</code> for non-paralyzable and "
        "<code>p&middot;e<sup>&minus;&lambda;(&tau;&minus;1)</sup></code> for paralyzable, with "
        "<code>p = 1&minus;e<sup>&minus;&lambda;</sup></code>.",
        "This is the load-bearing figure. Every other check either shares code with the "
        "simulator or only constrains the shape of the answer; these formulas share nothing "
        "with it and fix the absolute scale.",
        "Both models track their closed forms to <strong>3.7%</strong> across every "
        "statistically meaningful flux level. Non-paralyzable saturates at exactly "
        "<strong>1/&tau;</strong> (0.0996 measured vs 0.1000); paralyzable peaks and then "
        "collapses toward zero.",
        "Points drifting off the analytic curve as flux rises &mdash; that is the signature of a "
        "mis-set dead window. A constant offset instead would indicate an off-by-one in the "
        "blocked-bin count, since <code>&tau;</code> and <code>&tau;&minus;1</code> differ "
        "measurably at small &tau;. The three high-&lambda; paralyzable points sitting above the "
        "curve are <em>not</em> a defect: the analytic rate there is ~1e&minus;13, below the "
        "single-detection noise floor drawn on the plot, so they are excluded from the ratio panel.",
    ),
    (
        "11_deadtime_gaps.png",
        "Inter-detection gaps",
        "Free-running with a deliberately flat &phi;, at dead times of 1, 10 and 40 bins. "
        "Histogram of the gap between consecutive detections in absolute bins. Flat &phi; is "
        "essential &mdash; a structured &phi; imprints the laser period on this distribution and "
        "confounds it.",
        "The most diagnostic single plot for dead-time correctness. Three independent "
        "properties are visible at once, and each fails differently.",
        "A hard floor exactly at the dead time, a peak exactly there, and a geometric tail. "
        "The tail&rsquo;s decay rate is <code>e<sup>&minus;&lambda;</sup></code> per bin, so it "
        "<strong>recovers &lambda;</strong>: true 0.03, measured "
        "<strong>0.0300 / 0.0303 / 0.0299</strong>.",
        "Any count left of the red line means dead time is not being enforced. A peak displaced "
        "from the line by one bin means an off-by-one in the arm-time comparison. A curved "
        "rather than straight log-y tail means arrivals are not Poisson &mdash; which is exactly "
        "what a Bernoulli-per-bin sampler would produce at high flux.",
    ),
    (
        "13_pileup_coates.png",
        "Pile-up distortion and Coates inversion",
        "&phi; held deliberately <em>flat</em> and gated single-hit run at four flux levels, so "
        "every feature in the measured histogram is pile-up rather than scene structure. The "
        "Coates estimator is then applied to invert it.",
        "A round-trip test: distortion the simulator produces must be exactly removable by an "
        "estimator derived independently of it. Getting &phi; back is far stronger evidence "
        "than the raw histogram merely looking plausible.",
        "The raw histogram decays exponentially &mdash; up to <strong>100%</strong> between first "
        "and last bin &mdash; purely because early bins consume cycles later bins never get to "
        "see. Coates recovers flat &phi; at every flux, worst bin <strong>2.1&sigma;</strong> of "
        "counting noise.",
        "If the simulator over- or under-counted, Coates would return a sloped line rather than a "
        "flat one, and the slope would grow with flux. Dotted segments mark bins where fewer "
        "than 5% of cycles are still alive; the estimate is genuinely unreliable there and that "
        "is a property of the measurement, not a bug.",
    ),
    (
        "10_timestamp_raster.png",
        "Raw (cycle, bin) timestamps",
        "300 laser cycles at a dead time of 60 bins, scattered as one point per detection, under "
        "all three modes. &phi; is shown in its own panel rather than overlaid, where it would "
        "read as data.",
        "The simulator&rsquo;s primary output is timestamps; the histogram is a reduction of them. "
        "Looking at the raw object catches structural mistakes that survive histogramming.",
        "Gated single-hit yields <strong>at most one</strong> detection per cycle (251 total). "
        "Gated multi-hit and free-running allow several (345 and 339, max 3/cycle). Both show "
        "the return as a dense band at bin&nbsp;120 on a diffuse ambient background.",
        "Vertical striping would mean cycles are correlated when they should be independent. A "
        "second detection per cycle in the left panel would break the single-hit contract. "
        "Free-running and gated multi-hit look nearly identical here <em>by design</em> &mdash; "
        "they differ only within <code>dt</code> of a cycle boundary, worth 2.77% over 20,000 "
        "cycles, which is why that difference is asserted numerically rather than by eye.",
    ),
    (
        "14_mode_comparison.png",
        "The same &phi; through four operating modes",
        "One &phi; (a Gaussian return at 9&nbsp;m on an ambient floor) run through zero dead time, "
        "free-running, gated multi-hit and gated single-hit, shown both absolutely and "
        "area-normalised.",
        "Isolates what mode selection alone does to a measurement, with the scene held fixed. "
        "The zero-dead-time curve doubles as a direct test that detections reproduce &phi;.",
        "At <code>dt=0</code> the measured rate reproduces &phi; itself to <strong>2.9&sigma;</strong> "
        "of Poisson noise &mdash; confirming every arriving photon is detected. Dead time strictly "
        "reduces the total (1.754 vs 2.505). Gated single-hit caps at <strong>0.918</strong> "
        "detections per cycle and visibly biases the shape early.",
        "If <code>dt=0</code> did not reproduce &phi;, the sampler would be capping detections per "
        "bin &mdash; the Bernoulli failure mode this model was chosen to avoid. Free-running "
        "(thick green) and gated multi-hit (thin orange) are drawn at different widths precisely "
        "so their near-coincidence is visible as a halo rather than one curve hiding the other.",
    ),
    (
        "15_deadtime_over_cycle.png",
        "Dead time longer than one laser cycle",
        "Free-running at dead times of 0.5&times;, 1&times;, 2.5&times; and 5&times; the laser "
        "period, with the resulting histograms and total detection rates.",
        "A direct regression guard on <code>M3</code>. The previous implementation carried dead "
        "time in a buffer that reached back exactly one cycle, so anything longer silently "
        "saturated at one cycle instead of being applied.",
        "The rate keeps falling well past one cycle &mdash; <strong>1.425 &rarr; 0.849 &rarr; "
        "0.370 &rarr; 0.193</strong>, tracking 1/dt &mdash; and the gap floor is respected "
        "throughout. Tracking arm time in <em>absolute</em> bins removes the one-cycle lookback "
        "limit by construction rather than patching it.",
        "The old bug would show as the last two bars being equal to the 1&times; bar: dead time "
        "beyond one cycle having no further effect. Any detection pair closer than the dead time "
        "would mean the wrap is mishandled at a cycle boundary.",
    ),
    (
        "16_depth_bias.png",
        "Consequence for the measurement: depth bias vs flux",
        "A return at a known depth, gated single-hit, swept over two decades of flux. Depth is "
        "read out as the histogram <em>centroid</em>, not its argmax &mdash; argmax jitters by "
        "&plusmn;2 bins on shot noise alone at a few hundred detections, which swamps the effect.",
        "Connects the detection model to the quantity actually being measured, and separates two "
        "failure modes that are easily conflated.",
        "Uncorrected ranging degrades from <strong>&minus;0.09&nbsp;m to &minus;5.92&nbsp;m</strong> "
        "as flux rises. Coates is exact &mdash; worst <strong>+0.034&nbsp;m</strong>, under half a "
        "bin &mdash; while the whole cycle stays observable, and still holds to "
        "<strong>+0.064&nbsp;m</strong> against the truth restricted to whatever window remains, "
        "down to 40% visibility.",
        "The residual error above <strong>3.80 photons/cycle</strong> is <em>not</em> an "
        "implementation error and not something Coates can fix: past that flux the far part of "
        "the cycle is never sampled at all, so it is missing data rather than distortion. The "
        "green curve isolates this &mdash; it stays flat at zero throughout, showing the "
        "correction itself never degrades.",
    ),
]

# --------------------------------------------------------------------------- #
# Defect status                                                                #
# --------------------------------------------------------------------------- #
DEFECTS = [
    ("M2", "<code>n_hist_bins != n_tbins</code> raised instead of rebinning", "fixed",
     "The histogram is now a reduction of the timestamps, so a coarser histogram merges "
     "adjacent bins. Non-integral ratios raise with an explicit message rather than "
     "producing a silently wrong result."),
    ("M3", "Dead time longer than one cycle silently saturated at one cycle", "fixed",
     "Arm time is tracked in absolute bins instead of a one-cycle buffer, so the limit "
     "cannot exist. Demonstrated in figure&nbsp;6 and pinned by an exhaustive test over "
     "4 cycles at <code>dt = 2&times;</code> the period."),
    ("dt=0", "Closed-form async returned &prop;&phi; while &ldquo;MC truth&rdquo; returned "
             "&prop;1&minus;e<sup>&minus;&phi;</sup>", "resolved",
     "The verdict blamed the wrong component. That MC reference was "
     "<code>simulate_pixel_ewh</code>, whose Bernoulli-per-bin sampler caps detections at one "
     "per bin. With Poisson counts the ground truth returns &prop;&phi; and matches the closed "
     "form to <strong>2.3e&minus;4</strong>, against 2.9e&minus;2 for the old sampler. The "
     "closed form was right all along."),
    ("M5", "0.231&nbsp;s per pixel from nested Python loops", "improved",
     "Sparse arrival sampling via the Poisson superposition property &mdash; exact, not an "
     "approximation &mdash; gives <strong>565&times;</strong> at the nominal operating flux. "
     "The scan now visits only occupied bins, so cost is O(photons) rather than "
     "O(cycles&nbsp;&times;&nbsp;bins)."),
    ("M1", "Async closed form differs from Monte-Carlo by ~0.045", "open",
     "Unchanged and still unexplained. The obvious hypothesis &mdash; that the Bernoulli "
     "sampler imposed an implicit one-bin dead time, so <code>MC(dt=k)</code> should match "
     "<code>closed-form(dt=k+1)</code> &mdash; was tested and <strong>refuted</strong>: "
     "<code>dt+1</code> was worse in all three cases tried. Whether the new ground truth "
     "changes this has not yet been measured."),
]

# --------------------------------------------------------------------------- #
# Honest limits                                                                #
# --------------------------------------------------------------------------- #
LIMITS = [
    ("Sub-bin dead time is not representable",
     "A deliberate consequence of the bin-Poisson model, chosen because dead time in practice "
     "is either &ge; one bin or exactly zero. Dead times between those are outside the model, "
     "not merely untested. Non-integral <code>dt/&Delta;t</code> would need rounding."),
    ("Single pixel only",
     "Everything here simulates one pixel. Full-image work needs batched sampling and a "
     "chunked reduction &mdash; at 65,536&nbsp;px &times; 10k pulses the timestamp array alone "
     "is ~1.9&nbsp;GB, so the binding constraint is memory rather than compute."),
    ("Detector jitter is not modelled here",
     "The IRF is convolved into &phi; <em>before</em> detection. Real TDC jitter is a "
     "post-detection effect and belongs at a different point in the chain. Deferred by "
     "explicit decision."),
    ("Afterpulsing, crosstalk and PDE are absent",
     "The model covers arrival statistics, gating and dead time. Other SPAD non-idealities "
     "are simply not represented, so agreement here says nothing about them."),
    ("<code>simulate_pixel_ewh</code> is untouched",
     "The old Bernoulli sampler still exists and still backs the four remaining "
     "<code>xfail</code>s in <code>test_aspc_histogrammer.py</code>. Those tests accurately "
     "describe the old implementation; re-aiming them at the new ground truth is a pending "
     "decision, not an oversight."),
]

# --------------------------------------------------------------------------- #
# Checks that were wrong, not the code                                         #
# --------------------------------------------------------------------------- #
FALSE_ALARMS = [
    ("<code>argmax</code> depth readout looked biased at every flux",
     "Shot noise. At 648 detections spread over a 4-bin Gaussian the peak jitters &plusmn;2 "
     "bins. Switching to a centroid made the real bias appear cleanly: 7.79&nbsp;&rarr;&nbsp;2.08&nbsp;m."),
    ("Gap distribution appeared to peak away from the dead time",
     "The check clipped gaps before taking the mode, piling every large gap into the top bin "
     "and making that the mode. Raw counts peak exactly at <code>dt</code>."),
    ("Coates appeared to fail by 10.7%",
     "At the <em>lowest</em> flux, where a bin holds ~300 detections and shot noise alone is "
     "5.8% &mdash; so 10.7% is 1.9&sigma;. Now judged against counting noise rather than a flat "
     "percentage."),
    ("Paralyzable rate appeared to deviate by 6&times;10<sup>8</sup>%",
     "Dividing by an analytic rate of ~1e&minus;13, below what a finite simulation can "
     "resolve. The comparison is now restricted to flux levels where the expected count is "
     "large enough to mean anything."),
    ("A spike appeared at the right edge of every gap histogram",
     "<code>hist</code> closes its final bin at both ends, merging two gap values into one bar "
     "(334+343=677). Raw counts decay smoothly. Now plotted directly."),
    ("Coates appeared to fail by &minus;4.9&nbsp;m at high flux",
     "Genuine, but not a correction failure: at 40% window visibility the centroid over the "
     "observable region is necessarily early. Split into distortion (Coates fixes) and "
     "truncation (nothing does)."),
]

BADGES = {
    "fixed": ("fixed", "ok"),
    "resolved": ("resolved", "ok"),
    "improved": ("improved", "ok"),
    "open": ("open", "warn"),
}

EXTRA_CSS = """
.verdict{background:var(--teal-soft);border:1px solid var(--teal);border-radius:14px;
  padding:26px 30px;margin:36px 0 8px}
.verdict h2{margin:0 0 10px;font-size:1.45rem;color:var(--teal);letter-spacing:-.01em}
.verdict p{margin:0;color:var(--fg-2);max-width:78ch}
.tier{border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;
  background:var(--surface);padding:18px 22px;margin:14px 0}
.tier.strong{border-left-color:var(--teal)}
.tier.medium{border-left-color:var(--amber)}
.tier.weak{border-left-color:var(--fg-3)}
.tier h4{margin:0 0 4px;font-size:1.05rem}
.tier .tdesc{color:var(--fg-3);font-size:.9rem;margin:0 0 12px;max-width:74ch}
.tier ul{margin:0;padding-left:20px}
.tier li{margin:5px 0;color:var(--fg-2);font-size:.94rem}
.tierlabel{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
  font-weight:700;padding:2px 8px;border-radius:999px;margin-left:8px;vertical-align:middle}
.tierlabel.strong{background:var(--teal-soft);color:var(--teal)}
.tierlabel.medium{background:var(--amber-soft);color:var(--amber)}
.tierlabel.weak{background:var(--surface-2);color:var(--fg-3)}
.broken{background:var(--rust-soft);border-radius:8px;padding:12px 16px;margin-top:6px}
.broken .note-k{color:var(--rust)}
.limit{border-top:1px solid var(--line);padding:14px 0}
.limit b{display:block;margin-bottom:3px}
.limit span{color:var(--fg-3);font-size:.94rem}
.fa{display:grid;grid-template-columns:minmax(220px,1fr) 2fr;gap:14px 26px;
  border-top:1px solid var(--line);padding:14px 0;align-items:start}
.fa b{color:var(--fg-2);font-weight:600}
.fa span{color:var(--fg-3);font-size:.94rem}
@media(max-width:720px){.fa{grid-template-columns:1fr;gap:4px}}
.ck.fail{color:var(--rust)}
"""


def data_uri(path: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def read_checks(path: pathlib.Path):
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run aspc_detection_groundtruth.py first"
        )
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return rows


def build(figdir: pathlib.Path, checks_path: pathlib.Path) -> str:
    checks = read_checks(checks_path)
    n_pass = sum(1 for s, *_ in checks if s == "PASS")
    n_fail = len(checks) - n_pass

    tiers = ""
    for cls, name, desc, items in TIERS:
        lis = "".join(f"<li>{i}</li>" for i in items)
        tiers += (
            f'<div class="tier {cls}"><h4>{name}'
            f'<span class="tierlabel {cls}">{cls}</span></h4>'
            f'<p class="tdesc">{desc}</p><ul>{lis}</ul></div>'
        )

    figs = ""
    for i, (fname, title, howgen, why, found, broken) in enumerate(FIGS, 1):
        path = figdir / fname
        if not path.exists():
            raise SystemExit(
                f"missing figure {path} — run aspc_detection_groundtruth.py first"
            )
        figs += f"""
<section class="fig">
  <div class="fig-head"><span class="eyebrow">Figure {i}</span><h3>{title}</h3></div>
  <figure><img src="{data_uri(path)}" alt="{title}" />
    <figcaption>{fname}</figcaption></figure>
  <div class="fig-body">
    <div class="note"><span class="note-k">How it&rsquo;s generated</span><p>{howgen}</p></div>
    <div class="note"><span class="note-k">Why this plot</span><p>{why}</p></div>
    <div class="note"><span class="note-k">What it shows</span><p>{found}</p></div>
    <div class="note broken"><span class="note-k">What a bug would look like here</span><p>{broken}</p></div>
  </div>
</section>"""

    rows = ""
    for fid, prob, st, note in DEFECTS:
        label, cls = BADGES[st]
        rows += (
            f'<tr><td class="id"><code>{fid}</code></td><td>{prob}</td>'
            f'<td><span class="pill {cls}">{label}</span>'
            f'<span class="fix">{note}</span></td></tr>'
        )

    check_items = "".join(
        f'<li><span class="ck {"fail" if s != "PASS" else ""}">{s}</span><span class="ck-n">{n}</span>'
        f'<span class="ck-v">{v}</span></li>'
        for s, n, v in checks
    )
    limits = "".join(f'<div class="limit"><b>{t}</b><span>{d}</span></div>' for t, d in LIMITS)
    alarms = "".join(f'<div class="fa"><b>{t}</b><span>{d}</span></div>' for t, d in FALSE_ALARMS)
    stamp = _dt.date.today().isoformat()
    headline, detail = VERDICT

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Detection Model Verification</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">

<header class="top"><div class="inner">
  <span class="eyebrow">visionsim &middot; active SPC &middot; {stamp}</span>
  <h1>Ground-truth detection model</h1>
  <p class="lede measure">What happens to photon arrivals after &phi;: which of them become
  detections, under synchronous and free-running operation, with paralyzable and
  non-paralyzable dead time. Generated by
  <code>examples/sensors/aspc_detection_groundtruth.py</code> &mdash;
  <strong>{n_pass}/{len(checks)}</strong> self-checks passing, alongside 106 unit tests in
  <code>tests/test_aspc_detector.py</code>.</p>
</div></header>

<div class="verdict">
  <h2>{headline}</h2>
  <p>{detail}</p>
</div>

<section>
  <h2>How much the evidence is worth</h2>
  <p class="measure">A simulator that is consistently wrong can still pass most tests written
  against it. What separates real validation from self-agreement is whether the reference
  shares assumptions with the thing it checks &mdash; so the evidence below is graded by
  independence rather than counted.</p>
  {tiers}
</section>

<section>
  <h2>Figures</h2>
  <p class="measure">Ordered by how much they constrain correctness, not by the order the
  script emits them. Each carries an explicit description of how a broken implementation
  would present, so the figures can be read as diagnostics rather than decoration.</p>
  {figs}
</section>

<section>
  <h2>Defect status</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>ID</th><th>Problem</th><th>Status &amp; resolution</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
</section>

<section>
  <h2>What this does <em>not</em> establish</h2>
  <p class="measure">Scope limits that agreement above says nothing about. Several are
  deliberate model choices rather than gaps.</p>
  {limits}
</section>

<section>
  <h2>Failures that turned out to be bad checks</h2>
  <p class="measure">Recorded because they are the most likely source of future confusion: in
  each case the check was wrong and the simulator was right. Anyone re-running this and
  seeing a red line should suspect the test before the code.</p>
  {alarms}
</section>

<section>
  <h2>Self-checks from this run</h2>
  <p class="measure">Read directly from <code>detection_checks.tsv</code>, written by the run
  that produced the figures above, so these cannot drift from what was actually measured.
  {n_fail} failing.</p>
  <ul class="checks">{check_items}</ul>
</section>

<footer>
  <p>Reproduce with <code>PYTHONPATH=. python examples/sensors/aspc_detection_groundtruth.py</code>
  then <code>python examples/sensors/build_aspc_detection_report.py</code>.
  Companion report for the &phi; pipeline: <code>aspc_report.html</code>.</p>
</footer>

</div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = pathlib.Path(__file__).parent / "aspc"
    ap.add_argument("--figdir", type=pathlib.Path, default=here / "figures")
    ap.add_argument("--checks", type=pathlib.Path, default=here / "detection_checks.tsv")
    ap.add_argument("--out", type=pathlib.Path, default=here / "aspc_detection_report.html")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = build(args.figdir, args.checks)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out.resolve()}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

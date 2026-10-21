Optimizing the Keyframe Rate
============================

:doc:`Keyframing <keyframing>` explains the ``keyframe-multiplier``, the factor that stretches every keyframe to slow a motion down or speed it up. Picking it by hand means rendering a probe, estimating how far the motion moves between frames, and adjusting until it looks right. :func:`blender.optimize-rate <visionsim.cli.blender.optimize_rate>` searches for that multiplier so the typical inter-frame motion lands on a target.

|

The Problem
-----------

Motion between adjacent frames is what most downstream models see. If the camera moves too fast, adjacent frames have little in common and optical flow is large; if it moves too slowly, frames are near-duplicates. The multiplier ``k`` from :doc:`keyframing` scales this motion, but the right value depends on the scene, the trajectory and the render framerate.

``optimize-rate`` answers one question: which ``k`` makes the 95th-percentile optical flow magnitude about ``target`` pixels? The percentile is what keeps a few fast pixels, a frame with a depth discontinuity, or a noisy corner from setting the rate for the whole sequence.

|

How It Works
------------

The command runs a simplified Newton search. Optical flow scales roughly with ``1/k``, since a larger multiplier spreads the same motion over more frames, so the magnitude at a given ``k`` predicts the magnitude at the next one without a full re-render of the sequence. Each iteration:

1. Renders the animation at ``resolution-percentage`` of the scene resolution with only optical flow enabled, one sample per pixel, no denoising and no previews. The internal probe config forces these settings, so pass only the render options the probe still needs, such as ``--config.executable``.
2. Reads the flow magnitudes from the rendered EXRs and scales them to full resolution by ``100 / resolution-percentage``.
3. Feeds the magnitudes into a t-digest and reads back the ``percentile``-th value.
4. Stops when ``abs(p_flow - target) <= tolerance``, and returns ``k``.
5. Otherwise scales ``k`` toward the ratio ``p_flow / target``, clipping that ratio to ``max-scale-factor``, and damps the step by ``scale-decay`` raised to the iteration index to prevent oscillation.

The search runs for at most ``max-iterations`` steps. It also stops early, returning the current estimate with a warning, if two consecutive iterations change the percentile flow by less than ``stall-tolerance * target``, or if a probe render fails.

This assumes flow is proportional to ``1/k``. That holds closely for a uniform camera motion but less so near occlusions, so the search is a starting point to sanity-check rather than a measurement. Blender must be available for the probe renders, and the returned value is what you pass to the real render as ``--config.keyframe-multiplier``. The :doc:`quick-start <../../quick-start>` covers the ``render-animation`` flags it shares.

|

Parameters
----------

Positional
    ``PATH``
        Path to the blend file. Required.

Frames
    ``--frame-start``, ``--frame-end``, ``--frame-step``
        Restrict the search to a frame range, as for :func:`blender.render-animation <visionsim.cli.blender.render_animation>`.

Probe
    ``--resolution-percentage`` (default ``10``)
        Probe render resolution as a percentage of the scene resolution. Coarser is faster and noisier.
    ``--percentile`` (default ``95.0``)
        Percentile of the flow magnitudes to track.
    ``--target`` (default ``1.0``)
        Target flow magnitude in full-resolution pixels.
    ``--tolerance`` (default ``0.1``)
        Stop once the percentile flow is within this many pixels of ``target``.

Search
    ``--init-k`` (default ``5.0``)
        Initial keyframe multiplier. Must be positive.
    ``--max-iterations`` (default ``15``)
        Maximum number of probe renders.
    ``--max-scale-factor`` (default ``15.0``)
        Largest factor by which to adjust ``k`` in one step. Must be at least 1.
    ``--scale-decay`` (default ``0.95``)
        Multiplier applied to the step size after each iteration. Must be in (0, 1].
    ``--stall-tolerance`` (default ``0.05``)
        Stop early, with a warning, when the percentile flow moves by less than this fraction of ``target`` between iterations.

Depth and debug
    ``--max-depth`` (default unset)
        If set, ignore pixels farther than this depth. Depths are rendered for the probe when this is given.
    ``--debug-path`` (default unset)
        If set, write the per-iteration flow digest to ``debug-path / iter_II_k_KKKK.json``.

|

Usage
-----

Find a multiplier that puts the 95th-percentile flow near one pixel:

.. code-block:: bash

    visionsim blender.optimize-rate scene.blend --target 1.0 --percentile 95 \\
        --debug-path ./opt

The command logs the estimated rate and returns it. Render the final dataset with that multiplier:

.. code-block:: bash

    visionsim blender.render-animation scene.blend ./renders/scene \\
        --config.include-frames --config.include-flows --config.keyframe-multiplier 12.5

.. note::

    ``optimize-rate`` renders probes into a temporary directory and discards them. It never writes to the output directory of the final render, and nothing is persisted except the digest files under ``--debug-path`` when you pass it.

|

.. seealso::

    - :doc:`keyframing` for the keyframe multiplier and trajectory setup.
    - :doc:`../../quick-start` for the full ``render-animation`` workflow and flags.

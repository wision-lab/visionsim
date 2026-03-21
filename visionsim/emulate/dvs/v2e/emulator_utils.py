from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def lin_log(x: npt.NDArray, threshold: float = 20) -> npt.NDArray:
    """Apply a linear-or-logarithmic mapping to an intensity array.

    Below *threshold* the mapping is linear, above it the natural logarithm is
    used, keeping the two pieces smoothly joined.

    Args:
        x: Input intensity values in the range [0, 255].
        threshold: Transition point between the linear and log segments.

    Returns:
        The mapped values as a ``float32`` array of the same shape as *x*.
    """
    if x.dtype != np.float64:  # float64 needed for round-trip accuracy
        x = x.astype(np.float64)

    f = (1.0 / threshold) * math.log(threshold)
    with np.errstate(divide="ignore"):
        log_x = np.log(x)
    y = np.where(x <= threshold, x * f, log_x)

    # Round to a fixed precision so that adding and then subtracting the
    # threshold does not silently lose bits (which would suppress OFF events
    # that should follow ON events as a stimulus moves past).
    rounding = 1e8
    y = np.round(y * rounding) / rounding

    return y.astype(np.float32)


def rescale_intensity_frame(new_frame: npt.NDArray) -> npt.NDArray:
    """Rescale a raw intensity frame to the (0, 1] range used by filters.

    Applies the affine mapping ``(x + 20) / 275`` so that:
    * 0 maps to ~0.073 (prevents zero time-constants), and
    * 255 maps to ~1.0 (caps the maximum time-constant).

    Args:
        new_frame: Raw intensity frame with values in [0, 255].

    Returns:
        Rescaled frame in the approximate range (0.07, 1.0].
    """
    return (new_frame + 20) / 275.0


class LowPassFilter:
    """Stateful first-order IIR low-pass filter for log-intensity frames.

    Wraps the stateless filter computation and tracks the number of
    large-update warnings so that verbose output can be suppressed after a
    configurable limit.

    Args:
        max_warnings: Maximum number of large-eps warnings to emit before
            suppressing further messages.
    """

    def __init__(self, max_warnings: int = 3) -> None:
        self.max_warnings = max_warnings
        self.warning_count = 0

    def __call__(
        self,
        log_new_frame: npt.NDArray,
        lp_log_frame: npt.NDArray,
        inten01: npt.NDArray | None,
        delta_time: float,
        cutoff_hz: float = 0.0,
    ) -> npt.NDArray:
        """Apply an intensity-dependent first-order IIR low-pass filter.

        The time constant tau = 1 / (2 * pi * cutoff_hz) is scaled per-pixel by
        *inten01* so that brighter pixels respond faster.  When *cutoff_hz* <= 0
        the filter is bypassed and *log_new_frame* is returned unchanged.

        Args:
            log_new_frame: Current frame in lin-log representation.
            lp_log_frame: Previous filter state (same shape as *log_new_frame*).
            inten01: Per-pixel intensity scaling in (0, 1], or ``None`` to use a
                uniform time constant.
            delta_time: Elapsed time since the last frame, in seconds.
            cutoff_hz: 3 dB cutoff frequency of the filter in Hz.

        Returns:
            Updated filter state with the same shape as *log_new_frame*.
        """
        if cutoff_hz <= 0:
            return log_new_frame

        tau = 1 / (math.pi * 2 * cutoff_hz)

        # make the update proportional to the local intensity
        # the more intensity, the shorter the time constant
        if inten01 is not None:
            eps = inten01 * (delta_time / tau)
            max_eps = float(eps.max())
            if max_eps > 0.3 and self.warning_count < self.max_warnings:
                logger.warning(
                    f"IIR lowpass filter update has large maximum update "
                    f"eps={max_eps:.2f} from delta_time/tau={delta_time:.3g}/{tau:.3g}"
                )
                self.warning_count += 1
                if self.warning_count == self.max_warnings:
                    logger.warning(
                        "Suppressing further warnings about inaccurate IIR lowpass "
                        "filtering; check timestamp resolution and DVS photoreceptor "
                        "cutoff frequency"
                    )
            eps = np.minimum(eps, 1.0)  # keep filter stable
        else:
            eps = delta_time / tau

        return (1 - eps) * lp_log_frame + eps * log_new_frame


def subtract_leak_current(
    base_log_frame: npt.NDArray,
    leak_rate_hz: float,
    delta_time: float,
    pos_thres: float | npt.NDArray,
    leak_jitter_fraction: float,
    noise_rate_array: npt.NDArray,
) -> npt.NDArray:
    """Subtract a jittered leak current from the memorised log-intensity frame.

    Models junction leakage in the DVS reset switch, which manifests as a
    slow drift of the reference level and produces spurious ON events.

    Args:
        base_log_frame: Memorised log-intensity values at the change detector.
        leak_rate_hz: Nominal leak-event rate per pixel in Hz.
        delta_time: Elapsed time since the last frame, in seconds.
        pos_thres: Per-pixel ON threshold values (log units).
        leak_jitter_fraction: Fractional random variation applied to the leak
            rate to model per-pixel mismatch.
        noise_rate_array: Per-pixel log-normal noise-rate multipliers
            (pre-computed in ``_init``).

    Returns:
        Updated ``base_log_frame`` with the leak subtracted.
    """
    rand = np.random.randn(*noise_rate_array.shape).astype(np.float32)
    curr_leak_rate = leak_rate_hz * noise_rate_array * (1 - leak_jitter_fraction * rand)
    delta_leak = delta_time * curr_leak_rate * pos_thres
    return base_log_frame - delta_leak


def compute_event_map(
    diff_frame: npt.NDArray,
    pos_thres: float | npt.NDArray,
    neg_thres: float | npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Quantise a log-intensity difference frame into ON and OFF event counts.

    Each pixel accumulates one event per threshold multiple crossed during the
    inter-frame interval.

    Args:
        diff_frame: Log-intensity difference between the current photoreceptor
            output and the memorised reference, shape ``[H, W]``.
        pos_thres: Per-pixel ON threshold values, shape ``[H, W]``.
        neg_thres: Per-pixel OFF threshold values, shape ``[H, W]``.

    Returns:
        A 2-tuple ``(pos_evts_frame, neg_evts_frame)`` where each element is
        an ``int32`` array of shape ``[H, W]`` containing the number of ON
        and OFF events generated per pixel.
    """
    pos_frame = np.maximum(diff_frame, 0)
    neg_frame = np.maximum(-diff_frame, 0)

    pos_evts_frame = np.floor_divide(pos_frame, pos_thres).astype(np.int32)
    neg_evts_frame = np.floor_divide(neg_frame, neg_thres).astype(np.int32)

    return pos_evts_frame, neg_evts_frame


class PhotoreceptorNoiseVoltageComputer:
    """Cached computation of the photoreceptor RMS noise voltage.

    The photoreceptor injects Gaussian noise into the log-intensity signal that
    is subsequently first-order low-pass filtered with cutoff *f3db*.  This
    class inverts the relationship between the injected RMS amplitude and the
    resulting event rate so that callers can set the amplitude that produces
    a target ``shot_noise_rate_hz`` events per pixel per second at low light.

    The underlying empirical curve fit is described in:
    *Graca, R. & Delbruck, T. (2021). "Unraveling the Paradox of
    Intensity-Dependent DVS Pixel Noise." arXiv:2109.08640*.

    Results are cached: if *sample_rate_hz* changes by less than 10 % relative
    to the previous call, the cached value is returned immediately.
    """

    def __init__(self) -> None:
        self._last_sample_rate: float | None = None
        self._last_vn: float | None = None
        self._printed: bool = False

    def __call__(
        self,
        shot_noise_rate_hz: float,
        f3db: float,
        sample_rate_hz: float,
        pos_thr: float,
        neg_thr: float,
        sigma_thr: float,
    ) -> float:
        """Compute (or return cached) RMS noise voltage for the given parameters.

        Args:
            shot_noise_rate_hz: Target shot-noise event rate per pixel in Hz.
            f3db: First-order IIR RC lowpass filter 3 dB cutoff frequency in Hz.
            sample_rate_hz: Frame (sample) rate at which the noise is injected
                before IIR filtering, in Hz.
            pos_thr: Nominal ON threshold in natural-log units.
            neg_thr: Nominal OFF threshold in natural-log units.
            sigma_thr: Standard deviation of the threshold distribution in ln units.

        Returns:
            The Gaussian RMS noise amplitude in log_e units that should be added
            directly to the log photoreceptor output *before* low-pass filtering.
        """
        # Return cached result when sample rate has not changed significantly.
        if self._last_sample_rate is not None and self._last_vn is not None:
            diff = np.abs(sample_rate_hz / self._last_sample_rate - 1)
            if diff < 0.1:
                return self._last_vn

        # Normalise by bandwidth; simulation data cover ON-event rates, so divide
        # by 2 to recover the total (ON + OFF) rate.
        rate_per_bw = (shot_noise_rate_hz / f3db) / 2
        if rate_per_bw > 0.5:
            logger.warning(
                f"shot noise rate per hz of bandwidth is larger than 0.1 "
                f"(rate_hz={shot_noise_rate_hz} Hz, 3dB bandwidth={f3db} Hz)"
            )
        x = math.log10(rate_per_bw)
        if x < -5.0:
            logger.warning(
                f"desired noise rate of {shot_noise_rate_hz}Hz is too low to accurately compute a threshold value"
            )
        elif x > 0.0:
            logger.warning(
                f"desired noise rate of {shot_noise_rate_hz}Hz is too large to accurately compute a threshold value"
            )

        # now we need to numerically estimate the required Vnrms given the thresholds and the sigma thresholds,
        # since the noise rate varies dramatically with threshold
        N = 300  # num samples
        pos_samps = pos_thr + sigma_thr * np.random.randn(N)
        neg_samps = neg_thr + sigma_thr * np.random.randn(N)
        thrs = np.vstack((pos_samps, neg_samps))
        mins = np.min(thrs, axis=0)
        vns = np.array([self._vn_from_log_rate_per_hz(mins[i], x) for i in range(N)])
        vn = float(np.mean(vns))

        # Find the scaling factor from white noise to compensate for the
        # IIR lowpass filter's noise-equivalent bandwidth (NEB).
        # Simulate the same RC filter used in the emulator and measure RMS
        # attenuation, then pre-scale *vn* to compensate.
        self._last_sample_rate = sample_rate_hz
        tau = 1 / (f3db * 2 * math.pi)
        dt = 1 / sample_rate_hz
        t = np.arange(0, 1000 * tau, dt)
        rin = vn * np.random.randn(*t.shape)
        rms_in = np.std(rin)
        eps = dt / tau
        eps_limit = 0.1
        if eps > eps_limit:
            logger.warning(
                f"\neps={eps:.3f} for IIR lowpass is >{eps_limit}, either reduce "
                f"timestep (currently {dt:.3f}s) or decrease cutoff_hz "
                f"(currently {f3db:.3f} Hz)"
                f"\n\tExpect the generated shot noise rate to be significantly "
                "lower than the desired rate."
                "\n\tConsider not using --photoreceptor_noise if you only want "
                "simple Poisson shot noise without temporal correlation."
            )
        rout = np.zeros_like(rin)
        for i in range(1, len(rin)):
            rout[i] = rout[i - 1] * (1 - eps) + rin[i] * eps
        rms_out = np.std(rout)
        scale = rms_in / rms_out
        vnscaled = float(scale * vn)

        self._last_vn = vnscaled
        if not self._printed:
            logger.info(
                f"For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f} in ln units,"
                f" scaled by factor {scale:.3f} to {vnscaled:.3f} before 1st-order lowpass with sample rate {sample_rate_hz:.3} Hz, "
                f"sample interval dt={dt * 1000:.3f} ms,"
                f", cutoff_hz={f3db} Hz, tau={tau * 1000:.3f} ms,  Rn/f3dB={rate_per_bw:.3g} Hz, "
                f" and nominal on/off threshold={pos_thr}/{neg_thr} +/- {sigma_thr:.3f} ln units."
            )
            self._printed = True
        return vnscaled

    @staticmethod
    def _vn_from_log_rate_per_hz(thr: float, x: float) -> float:
        """Return required noise RMS *vn* for a given threshold and log-rate ratio.

        Args:
            thr: Pixel threshold in ln units.
            x: ``log10(shot_noise_rate_hz / (2 * f3db))``; the operating point
               on the empirical curve.

        Returns:
            Required noise RMS amplitude *vn* in ln units.
        """
        # Empirical cubic polynomial fit (y = log10(thr / Vn) vs x):
        y = -0.0026 * x**3 - 0.036 * x**2 - 0.1949 * x + 0.321
        thr_per_vn = 10**y
        return thr / thr_per_vn


def generate_shot_noise(
    shot_noise_rate_hz: float,
    delta_time: float,
    shot_noise_inten_factor: float,
    inten01: npt.NDArray,
    pos_thres_pre_prob: float | npt.NDArray,
    neg_thres_pre_prob: float | npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate per-pixel shot-noise event masks for one inter-frame interval.

    A single uniform random draw per pixel determines whether an ON or OFF
    shot-noise event fires.  The per-pixel probabilities are modulated by
    intensity (brighter pixels have slightly more shot noise) and by the
    per-pixel threshold (lower threshold → higher event probability).

    Args:
        shot_noise_rate_hz: Nominal shot-noise event rate per pixel in Hz.
        delta_time: Duration of the current inter-frame interval in seconds.
        shot_noise_inten_factor: Multiplicative factor modelling the increase
            of shot noise with intensity (typically 0.25).
        inten01: Per-pixel intensities in the range [0, 1], shape ``[H, W]``.
        pos_thres_pre_prob: Per-pixel ON probability scaling factor equal to
            ``pos_thres_nominal / pos_thres``; shape ``[H, W]``.
        neg_thres_pre_prob: Per-pixel OFF probability scaling factor equal to
            ``neg_thres_nominal / neg_thres``; shape ``[H, W]``.

    Returns:
        A 2-tuple ``(shot_on_cord, shot_off_cord)`` of boolean arrays, each
        of shape ``[H, W]``, where ``True`` indicates a shot-noise event at
        that pixel.
    """
    if shot_noise_rate_hz * delta_time > 1:
        logger.warning(
            f"shot_noise_rate_hz*delta_time="
            f"{shot_noise_rate_hz:.2f}*{delta_time:.2g}="
            f"{shot_noise_rate_hz * delta_time:.2f} is too large; "
            "decrease timestamp resolution or sample rate"
        )

    # shot noise factor is the probability of generating an OFF event in this frame (which is tiny typically)
    # we compute it by taking half the total shot noise rate (OFF only),
    # multiplying by the delta time of this frame,
    # and multiplying by the intensity factor
    shot_noise_factor = ((shot_noise_rate_hz / 2) * delta_time) * ((shot_noise_inten_factor - 1) * inten01 + 1)

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    one_minus_shot_on_prob = 1 - shot_noise_factor * pos_thres_pre_prob
    shot_off_prob = shot_noise_factor * neg_thres_pre_prob

    # for shot noise generate rands from 0-1 for each pixel
    rand01 = np.random.random_sample(inten01.shape).astype(np.float32)

    # precompute all the shot noise cords, gets binary array size of chip
    shot_on_cord = rand01 > one_minus_shot_on_prob
    shot_off_cord = rand01 < shot_off_prob

    return shot_on_cord, shot_off_cord


# Discrete Laplacian kernel used for the centre-surround resistive diffuser.
CSDVS_LAPLACIAN_KERNEL: np.ndarray = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

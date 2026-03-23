from __future__ import annotations

import logging
import math
import sys
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve as ndimage_convolve

from .emulator_utils import (
    CSDVS_LAPLACIAN_KERNEL,
    LowPassFilter,
    PhotoreceptorNoiseVoltageComputer,
    compute_event_map,
    generate_shot_noise,
    lin_log,
    rescale_intensity_frame,
    subtract_leak_current,
)

logger = logging.getLogger(__name__)


class EventEmulator:
    """Compute DVS events from a sequence of intensity frames.

    Implements a behavioural model of a Dynamic Vision Sensor (DVS) pixel
    array, including:

    * Log-domain contrast-change detection with per-pixel threshold mismatch.
    * Intensity-dependent first-order low-pass photoreceptor filtering.
    * Leak current simulation (slow baseline drift -> spurious ON events).
    * Refractory period enforcement per pixel.
    * Simple Poisson shot noise or correlated photoreceptor-noise model.
    * Optional centre-surround (CSDVS) inhibition via a resistive-diffuser surround.
    * Optional SCIDVS non-linear high-pass amplified photoreceptor.
    """

    MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING: float = 1e-5
    SCIDVS_GAIN: float = 2  # gain after highpass
    SCIDVS_TAU_S: float = 0.01  # small-signal time constant in seconds

    # Log-normal sigma of the per-pixel SCIDVS time-constant distribution.
    SCIDVS_TAU_COV: float = 0.5

    # Models the slight increase of shot noise with intensity.
    SHOT_NOISE_INTEN_FACTOR: float = 0.25

    def __init__(
        self,
        pos_thres: float = 0.2,
        neg_thres: float = 0.2,
        sigma_thres: float = 0.03,
        cutoff_hz: float = 0.0,
        leak_rate_hz: float = 0.1,
        refractory_period_s: float = 0.0,
        shot_noise_rate_hz: float = 0.0,
        photoreceptor_noise: bool = False,
        leak_jitter_fraction: float = 0.1,
        noise_rate_cov_decades: float = 0.1,
        seed: int | None = None,
        cs_lambda_pixels: float | None = None,
        cs_tau_p_ms: float | None = None,
        hdr: bool = False,
        scidvs: bool = False,
    ) -> None:
        """Initialise the emulator with sensor and noise parameters.

        Args:
            pos_thres: Nominal ON-event threshold in log_e intensity units.
            neg_thres: Nominal OFF-event threshold in log_e intensity units.
            sigma_thres: Standard deviation of the per-pixel threshold
                distribution in log_e units.
            cutoff_hz: 3 dB cutoff frequency of the DVS photoreceptor low-pass
                filter in Hz.  Set to 0 to disable filtering.
            leak_rate_hz: Nominal leak-event rate per pixel in Hz, modelling
                junction leakage in the reset switch.
            refractory_period_s: Minimum inter-event interval per pixel in
                seconds.  Set to 0 to disable.
            shot_noise_rate_hz: Poisson shot-noise event rate per pixel in Hz.
                Set to 0 to disable.
            photoreceptor_noise: When ``True``, model shot noise as correlated
                Gaussian noise injected at the photoreceptor rather than as
                independent Poisson events.  Requires both *shot_noise_rate_hz*
                and *cutoff_hz* to be non-zero.
            leak_jitter_fraction: Fractional random variation applied to the
                per-pixel leak rate to model device mismatch.
            noise_rate_cov_decades: Standard deviation (in decades) of the
                log-normal per-pixel noise-rate distribution.
            seed: Random seed for threshold mismatch and noise.  Use ``None`` for a
                new random seed on every run.
            cs_lambda_pixels: Space constant of the centre-surround surround in
                pixels.  Pass ``None`` to disable surround inhibition.
            cs_tau_p_ms: Time constant of the surround low-pass filter in ms,
                or ``None`` / 0 for an instantaneous surround.
            hdr: When ``True``, treat the input as HDR floating-point
                log-encoded grey scale where pixel value 255 corresponds to
                ``ln(255) ≈ 5.54``.
            scidvs: When ``True``, simulate the high-gain adaptive
                photoreceptor of the SCIDVS pixel.
        """
        logger.info(f"ON/OFF log_e temporal contrast thresholds: {pos_thres} / {neg_thres} +/- {sigma_thres}")

        if photoreceptor_noise:
            if shot_noise_rate_hz == 0:
                logger.warning(
                    "Parameter `photoreceptor_noise` is specified but `shot_noise_rate_hz` is 0; "
                    "set a finite rate of shot noise events per pixel"
                )
                sys.exit(1)
            if cutoff_hz == 0:
                logger.warning(
                    "Parameter `photoreceptor_noise` is specified but `cutoff_hz` is 0; "
                    "set a finite photoreceptor cutoff frequency"
                )
                sys.exit(1)

        if seed is not None:
            np.random.seed(seed)

        # ------------------------------------------------------------------ #
        # Sensor parameters                                                  #
        # ------------------------------------------------------------------ #
        # pos_thres / neg_thres start as scalars and are replaced with
        # per-pixel arrays by _init() once the frame shape is known.
        self.pos_thres: float | npt.NDArray = pos_thres
        self.neg_thres: float | npt.NDArray = neg_thres
        self.pos_thres_nominal: float = pos_thres
        self.neg_thres_nominal: float = neg_thres
        self.sigma_thres = sigma_thres

        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.photoreceptor_noise = photoreceptor_noise

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.log_input = hdr
        self.scidvs = scidvs

        if self.log_input:
            logger.info("Treating input as log-encoded HDR input")

        if self.scidvs:
            logger.info("Modeling potential SCIDVS pixel with nonlinear CR highpass amplified log intensity")

        # ------------------------------------------------------------------ #
        # Centre-surround DVS (CSDVS) configuration                          #
        # ------------------------------------------------------------------ #
        self.cs_tau_p_ms = cs_tau_p_ms
        self.cs_lambda_pixels = cs_lambda_pixels
        self.csdvs_enabled = cs_lambda_pixels is not None
        self.cs_steps_warning_printed = False

        if cs_lambda_pixels is not None:
            self.cs_tau_h_ms: float = (
                0.0 if (cs_tau_p_ms is None or cs_tau_p_ms == 0) else cs_tau_p_ms / (cs_lambda_pixels**2)
            )
            lat_res = 1 / (cs_lambda_pixels**2)
            trans_cond = 1 / cs_lambda_pixels
            logger.debug(
                f"lateral resistance R={lat_res:.2g}Ohm, "
                f"transverse transconductance g={trans_cond:.2g} Siemens, "
                f"Rg={(lat_res * trans_cond):.2f}"
            )
            logger.info(
                f"Centre-surround parameters:\n\t"
                f"cs_tau_p_ms: {self.cs_tau_p_ms}\n\t"
                f"cs_tau_h_ms:  {self.cs_tau_h_ms}\n\t"
                f"cs_lambda_pixels:  {cs_lambda_pixels:.2f}\n\t"
            )

        # ------------------------------------------------------------------ #
        # Stateful utility objects (no global state)                         #
        # ------------------------------------------------------------------ #
        self._low_pass_filter = LowPassFilter()
        self._noise_vrms_computer = PhotoreceptorNoiseVoltageComputer()

        # ------------------------------------------------------------------ #
        # Runtime state – reset by reset(), lazily populated by _init()      #
        # ------------------------------------------------------------------ #
        # None until generate_events is first called.
        self.lp_log_frame: npt.NDArray | None = None
        self.cs_surround_frame: npt.NDArray | None = None
        self.base_log_frame: npt.NDArray | None = None

        # SCIDVS adaptive photoreceptor state.
        self.scidvs_highpass: npt.NDArray | None = None
        self.scidvs_previous_photo: npt.NDArray | None = None
        self.scidvs_tau_arr: npt.NDArray | None = None

        # Photoreceptor-noise model state.
        self.photoreceptor_noise_vrms: float | None = None
        self.photoreceptor_noise_arr: npt.NDArray | None = None

        # Per-pixel shot-noise probability factors (scalar until _init runs).
        self.pos_thres_pre_prob: float | npt.NDArray = 1.0
        self.neg_thres_pre_prob: float | npt.NDArray = 1.0

        # Leak noise-rate array (set by _init when leak_rate_hz > 0).
        self.noise_rate_array: npt.NDArray | None = None

        # Refractory-period timestamp memory (set by _init when refractory_period_s > 0).
        self.timestamp_mem: npt.NDArray | None = None
        self.reset()

    # ---------------------------------------------------------------------- #
    # Public interface                                                       #
    # ---------------------------------------------------------------------- #

    def reset(self) -> None:
        """Reset the emulator state.

        Clears all internal frame buffers and event counters so that the next
        call to :meth:`generate_events` will reinitialise the emulator as if
        it had just been constructed.
        """
        self.num_events_total: int = 0
        self.num_events_on: int = 0
        self.num_events_off: int = 0
        self.no_events_warning_count: int = 0
        self.frame_counter: int = 0
        self.t_previous: float = 0.0

        self.lp_log_frame = None
        self.cs_surround_frame = None
        self.base_log_frame = None
        self.scidvs_highpass = None
        self.scidvs_previous_photo = None
        self.scidvs_tau_arr = None
        self.photoreceptor_noise_vrms = None
        self.photoreceptor_noise_arr = None
        self.noise_rate_array = None
        self.timestamp_mem = None

    def use_preset(self, model: Literal["clean", "noisy"]) -> None:
        """Apply a named preset of DVS sensor parameters.

        Two presets are available:

        * ``"clean"`` – ideal sensor with no noise or mismatch.
        * ``"noisy"`` – realistic sensor with typical noise levels.

        Raises:
            ValueError: If the *model* name is not recognized.

        Args:
            model: Name of the parameter preset (``"clean"`` or ``"noisy"``).
        """
        if model == "clean":
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.02
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.leak_jitter_fraction = 0
            self.noise_rate_cov_decades = 0
            self.shot_noise_rate_hz = 0
            self.refractory_period_s = 0
        elif model == "noisy":
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.05
            self.cutoff_hz = 30
            self.leak_rate_hz = 0.1
            self.shot_noise_rate_hz = 5.0
            self.refractory_period_s = 0
            self.leak_jitter_fraction = 0.1
            self.noise_rate_cov_decades = 0.1
        else:
            raise ValueError(f"Unknown DVS model: {model}")

        logger.info(
            f"set DVS model params with option '{model}' to the following values:\n"
            f"pos_thres={self.pos_thres}\n"
            f"neg_thres={self.neg_thres}\n"
            f"sigma_thres={self.sigma_thres}\n"
            f"cutoff_hz={self.cutoff_hz}\n"
            f"leak_rate_hz={self.leak_rate_hz}\n"
            f"shot_noise_rate_hz={self.shot_noise_rate_hz}\n"
            f"refractory_period_s={self.refractory_period_s}"
        )

    def generate_events(
        self,
        new_frame: npt.NDArray,
        t_frame: float,
    ) -> npt.NDArray | None:
        """Generate DVS events from a new intensity frame.

        On the **first call** the emulator initialises its internal state and
        returns ``None`` (no events from a single frame).  Subsequent calls
        produce events based on the log-intensity change since the last frame.

        Args:
            new_frame: Intensity frame of shape ``[H, W]``.  Note that the
                first axis is *y* (rows) and the second is *x* (columns),
                matching the MATLAB / NumPy convention.
            t_frame: Timestamp of this frame in seconds.

        Raises:
            ValueError: If *t_frame* is earlier than the previous frame's
                timestamp.

        Returns:
            An ``[N, 4]`` ``float32`` array where each row is
            ``[timestamp, x, y, polarity]`` with polarity ``+1`` for ON events
            and ``-1`` for OFF events, or ``None`` if no events were generated.
        """
        self.frame_counter += 1

        if t_frame < self.t_previous:
            raise ValueError(f"this frame time={t_frame} must be later than previous frame time={self.t_previous}")

        delta_time = t_frame - self.t_previous

        if self.log_input and new_frame.dtype != np.float32:
            logger.warning("log_frame is True but input frame is not np.float32 datatype")

        new_frame_f64 = new_frame.astype(np.float64)
        log_new_frame = lin_log(new_frame_f64) if not self.log_input else new_frame_f64.astype(np.float32)
        inten01: npt.NDArray | None = None

        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:
            inten01 = rescale_intensity_frame(new_frame_f64.copy())

        # --- Initialise IIR state on the very first frame -------------------
        if self.lp_log_frame is None:
            self.lp_log_frame = log_new_frame.copy()
        if self.photoreceptor_noise_arr is None:
            self.photoreceptor_noise_arr = np.zeros_like(log_new_frame)

        lp_log_frame: npt.NDArray = self._low_pass_filter(
            log_new_frame=log_new_frame,
            lp_log_frame=self.lp_log_frame,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cutoff_hz,
        )
        self.lp_log_frame = lp_log_frame
        photoreceptor_noise_arr: npt.NDArray = self.photoreceptor_noise_arr

        # Photoreceptor-noise model: inject noise only after the base frame is
        # memorised so that the IIR filter has had time to settle.
        if self.photoreceptor_noise and self.base_log_frame is not None:
            self.photoreceptor_noise_vrms = self._noise_vrms_computer(
                shot_noise_rate_hz=self.shot_noise_rate_hz,
                f3db=self.cutoff_hz,
                sample_rate_hz=1 / delta_time,
                pos_thr=self.pos_thres_nominal,
                neg_thr=self.neg_thres_nominal,
                sigma_thr=self.sigma_thres,
            )
            noise = self.photoreceptor_noise_vrms * np.random.randn(*log_new_frame.shape).astype(np.float32)
            photoreceptor_noise_arr = self._low_pass_filter(
                noise, photoreceptor_noise_arr, None, delta_time, self.cutoff_hz
            )
            self.photoreceptor_noise_arr = photoreceptor_noise_arr

        # Surround computations by time-stepping the resistive diffuser.
        if cs_lambda_pixels := self.cs_lambda_pixels:
            self.cs_surround_frame = self._update_csdvs(lp_log_frame, cs_lambda_pixels, delta_time)

        # --- First-frame initialisation: set reference level, then return ---
        if self.base_log_frame is None:
            self._init(new_frame)
            if self.cs_surround_frame is None:
                self.base_log_frame = lp_log_frame.copy()
            else:
                self.base_log_frame = lp_log_frame - self.cs_surround_frame
            return None

        # From here on, base_log_frame is guaranteed non-None (mypy)
        base_log_frame: npt.NDArray = self.base_log_frame

        # --- SCIDVS high-pass filter update ---------------------------------
        if self.scidvs:
            if self.scidvs_highpass is None or self.scidvs_previous_photo is None:
                self.scidvs_highpass = np.zeros_like(lp_log_frame)
                self.scidvs_previous_photo = lp_log_frame.copy()
            scidvs_highpass: npt.NDArray = self.scidvs_highpass
            scidvs_previous_photo: npt.NDArray = self.scidvs_previous_photo
            scidvs_highpass = (
                scidvs_highpass
                + (lp_log_frame - scidvs_previous_photo)
                - delta_time * self._scidvs_dvdt(scidvs_highpass, self.scidvs_tau_arr)
            )
            self.scidvs_highpass = scidvs_highpass
            self.scidvs_previous_photo = lp_log_frame.copy()

        # --- Leak current ---------------------------------------------------
        # dI = R_l * Theta_on * dt  (drift of memorised reference)
        if self.leak_rate_hz > 0 and self.noise_rate_array is not None:
            base_log_frame = subtract_leak_current(
                base_log_frame=base_log_frame,
                leak_rate_hz=self.leak_rate_hz,
                delta_time=delta_time,
                pos_thres=self.pos_thres,
                leak_jitter_fraction=self.leak_jitter_fraction,
                noise_rate_array=self.noise_rate_array,
            )
            self.base_log_frame = base_log_frame

        # --- Select photoreceptor output ------------------------------------
        if self.scidvs and self.scidvs_highpass is not None:
            photoreceptor: npt.NDArray = EventEmulator.SCIDVS_GAIN * self.scidvs_highpass
        else:
            photoreceptor = lp_log_frame

        if self.cs_surround_frame is None:
            diff_frame = photoreceptor + photoreceptor_noise_arr - base_log_frame
        else:
            diff_frame = photoreceptor + photoreceptor_noise_arr - self.cs_surround_frame - base_log_frame

        pos_evts_frame, neg_evts_frame = compute_event_map(diff_frame, self.pos_thres, self.neg_thres)
        max_events_per_pixel: int = int(max(pos_evts_frame.max(), neg_evts_frame.max()))
        if max_events_per_pixel > 100:
            logger.warning(f"Too many events generated for this frame: num_iter={max_events_per_pixel}>100 events")

        # Preallocate the event collector array.
        events: npt.NDArray = np.empty((0, 4), dtype=np.float32)

        # Distribute events uniformly across the inter-frame interval.
        # Timestamps start one step after t_previous and end at t_frame.
        num_ts_steps = max_events_per_pixel if max_events_per_pixel > 0 else 1
        ts_step = delta_time / num_ts_steps
        timestamps = np.linspace(
            start=self.t_previous + ts_step,
            stop=t_frame,
            num=num_ts_steps,
            dtype=np.float32,
        )

        # Accumulators for base-frame update.
        final_pos_evts_frame = np.zeros(pos_evts_frame.shape, dtype=np.int32)
        final_neg_evts_frame = np.zeros(neg_evts_frame.shape, dtype=np.int32)

        if max_events_per_pixel == 0 and self.no_events_warning_count < 100:
            logger.warning(f"no signal events generated for frame #{self.frame_counter:,} at t={t_frame:.4f}s")
            self.no_events_warning_count += 1
            # events = np.empty((0, 4), dtype=np.float32) # This line is now redundant
        else:
            # OPTIMIZATION: Process events using sparse coordinates instead of scanning the whole 2D image
            # for every sub-step. This is much faster for high-resolution frames.
            pos_y, pos_x = np.nonzero(pos_evts_frame)
            pos_counts = pos_evts_frame[pos_y, pos_x]

            neg_y, neg_x = np.nonzero(neg_evts_frame)
            neg_counts = neg_evts_frame[neg_y, neg_x]

            events_list = []
            for i in range(max_events_per_pixel):
                level = i + 1
                # Sparse masks for this iteration
                mask_p = pos_counts >= level
                curr_y_p, curr_x_p = pos_y[mask_p], pos_x[mask_p]

                mask_n = neg_counts >= level
                curr_y_n, curr_x_n = neg_y[mask_n], neg_x[mask_n]

                if self.refractory_period_s > ts_step and (ts_mem := self.timestamp_mem) is not None:
                    # Sparse refractory check
                    t_last_p = ts_mem[curr_y_p, curr_x_p]
                    valid_p = (timestamps[i] - t_last_p) > self.refractory_period_s
                    curr_y_p, curr_x_p = curr_y_p[valid_p], curr_x_p[valid_p]
                    ts_mem[curr_y_p, curr_x_p] = timestamps[i]

                    t_last_n = ts_mem[curr_y_n, curr_x_n]
                    valid_n = (timestamps[i] - t_last_n) > self.refractory_period_s
                    curr_y_n, curr_x_n = curr_y_n[valid_n], curr_x_n[valid_n]
                    ts_mem[curr_y_n, curr_x_n] = timestamps[i]

                # Accumulate the count of events that actually fired (after refractory filtering)
                final_pos_evts_frame[curr_y_p, curr_x_p] += 1
                final_neg_evts_frame[curr_y_n, curr_x_n] += 1

                iter_ev = self._get_event_list_from_coords((curr_y_p, curr_x_p), (curr_y_n, curr_x_n), timestamps[i])
                if iter_ev is not None:
                    # Shuffle only the events occurring at the exact same timestamp
                    shuf = np.random.permutation(len(iter_ev))
                    events_list.append(iter_ev[shuf])

            # OPTIMIZATION: Single concatenation at the end is O(N) instead of O(N^2) in the loop.
            if events_list:
                events = np.concatenate(events_list)
            else:
                return None

        # Shot noise (simple Poisson model; not used with photoreceptor_noise).
        shot_on_cord: npt.NDArray | None = None

        # inten01 is guaranteed non-None here because shot_noise_rate_hz > 0
        # triggered its computation above.
        if self.shot_noise_rate_hz > 0 and not self.photoreceptor_noise and inten01 is not None:
            shot_on_cord, shot_off_cord = generate_shot_noise(
                shot_noise_rate_hz=self.shot_noise_rate_hz,
                delta_time=delta_time,
                shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
                inten01=inten01,
                pos_thres_pre_prob=self.pos_thres_pre_prob,
                neg_thres_pre_prob=self.neg_thres_pre_prob,
            )

            shot_on_xy = np.nonzero(shot_on_cord)
            shot_off_xy = np.nonzero(shot_off_cord)

            # Assign the last signal timestamp to all shot-noise events.
            shot_noise_events = self._get_event_list_from_coords(shot_on_xy, shot_off_xy, timestamps[-1])

            if shot_noise_events is not None:
                events = np.concatenate((events, shot_noise_events))

        # Update the memorised reference level with the emitted events.
        base_log_frame = base_log_frame + final_pos_evts_frame * self.pos_thres
        base_log_frame = base_log_frame - final_neg_evts_frame * self.neg_thres
        self.base_log_frame = base_log_frame

        if not self.photoreceptor_noise and self.shot_noise_rate_hz > 0 and shot_on_cord is not None:
            base_log_frame[shot_on_xy] = lp_log_frame[shot_on_xy]
            base_log_frame[shot_off_xy] = lp_log_frame[shot_off_xy]

        if len(events) > 0:
            if np.any(np.diff(events[:, 0]) < 0):
                logger.warning("nonmonotonic timestamp(s) detected in events")

        self.t_previous = t_frame
        return events

    # ---------------------------------------------------------------------- #
    # Private methods                                                        #
    # ---------------------------------------------------------------------- #

    def _init(self, first_frame_linear: npt.NDArray) -> None:
        """Initialise per-pixel data structures from the first input frame.

        Called once on the very first call to :meth:`generate_events`.  Sets up
        random per-pixel threshold arrays, the leak noise-rate map, and the
        refractory-period timestamp memory.

        Args:
            first_frame_linear: The first input frame (H × W) used only for
                its shape; pixel values are not used directly here.
        """
        logger.debug("Initialising random temporal contrast thresholds from base frame")

        # take the variance of threshold into account.
        if self.sigma_thres > 0:
            self.pos_thres = np.random.normal(self.pos_thres, self.sigma_thres, size=first_frame_linear.shape).astype(
                np.float32
            )
            # to avoid the situation where the threshold is too small.
            self.pos_thres = np.clip(self.pos_thres, a_min=0.01, a_max=None)

            self.neg_thres = np.random.normal(self.neg_thres, self.sigma_thres, size=first_frame_linear.shape).astype(
                np.float32
            )
            self.neg_thres = np.clip(self.neg_thres, a_min=0.01, a_max=None)

        # Pre-compute shot-noise probability scalings.
        self.pos_thres_pre_prob = self.pos_thres_nominal / self.pos_thres
        self.neg_thres_pre_prob = self.neg_thres_nominal / self.neg_thres

        if self.scidvs and EventEmulator.SCIDVS_TAU_COV > 0:
            self.scidvs_tau_arr = EventEmulator.SCIDVS_TAU_S * np.exp(
                np.random.normal(0, EventEmulator.SCIDVS_TAU_COV, size=first_frame_linear.shape).astype(np.float32)
            )

        # If leak is non-zero, initialise the noise-rate array now (log-normal distribution).
        # Doing this *after* we determine randomly distributed thresholds ensures that
        # low-threshold pixels don't generate a burst of events at the first frame.
        if self.leak_rate_hz > 0:
            noise_rate_array = np.random.randn(*first_frame_linear.shape).astype(np.float32)
            self.noise_rate_array = np.exp(math.log(10) * self.noise_rate_cov_decades * noise_rate_array)

        if self.refractory_period_s > 0:
            self.timestamp_mem = np.zeros(first_frame_linear.shape, dtype=np.float32) - self.refractory_period_s

    def _scidvs_dvdt(
        self,
        v: npt.NDArray,
        tau: npt.NDArray | float | None = None,
    ) -> npt.NDArray:
        """Compute the time derivative of the SCIDVS high-pass node voltage.

        The SCIDVS pixel uses a nonlinear conductance (sinh) to model the
        adaptive photoreceptor's high-pass characteristic.

        Args:
            v: Current high-pass node voltage (in log_e intensity units),
               shape ``[H, W]``.
            tau: Time constant in seconds.  Uses the class-level
                :attr:`SCIDVS_TAU_S` when ``None``.

        Returns:
            Time derivative ``dv/dt`` with the same shape as *v*.
        """
        if tau is None:
            tau = EventEmulator.SCIDVS_TAU_S
        efold = 1 / 0.7  # e-fold of sinh conductance in log_e units (≈ 1/κ)
        return (1 / tau) * np.sinh(v / efold)

    def _get_event_list_from_coords(
        self,
        pos_event_xy: tuple[npt.NDArray, ...],
        neg_event_xy: tuple[npt.NDArray, ...],
        timestamp: float,
    ) -> npt.NDArray | None:
        """Build a ``[N, 4]`` AER event array from ON and OFF pixel coordinates.

        Args:
            pos_event_xy: 2-tuple ``(y_coords, x_coords)`` of 1-D arrays
                listing the pixel addresses of ON events.
            neg_event_xy: 2-tuple ``(y_coords, x_coords)`` of 1-D arrays
                listing the pixel addresses of OFF events.
            timestamp: Scalar timestamp assigned to all events in this iteration.

        Returns:
            A ``float32`` array of shape ``[N, 4]`` with columns
            ``[timestamp, x, y, polarity]`` (polarity ``+1`` ON, ``-1`` OFF),
            or ``None`` if there are no events.
        """
        num_pos_events = pos_event_xy[0].shape[0]
        num_neg_events = neg_event_xy[0].shape[0]
        num_events = num_pos_events + num_neg_events

        if num_events == 0:
            return None

        self.num_events_on += num_pos_events
        self.num_events_off += num_neg_events
        self.num_events_total += num_events

        # Use np.empty for faster allocation (avoids pre-filling with ones)
        event_array = np.empty((num_events, 4), dtype=np.float32)
        event_array[:, 0] = timestamp

        # ON event coordinates (index 0 = y, index 1 = x).
        if num_pos_events > 0:
            event_array[:num_pos_events, 1] = pos_event_xy[1]
            event_array[:num_pos_events, 2] = pos_event_xy[0]
            event_array[:num_pos_events, 3] = 1.0

        # OFF event coordinates.
        if num_neg_events > 0:
            event_array[num_pos_events:, 1] = neg_event_xy[1]
            event_array[num_pos_events:, 2] = neg_event_xy[0]
            event_array[num_pos_events:, 3] = -1.0

        return event_array

    def _update_csdvs(
        self,
        lp_log_frame: npt.NDArray,
        cs_lambda_pixels: float,
        delta_time: float,
    ) -> npt.NDArray:
        """Advance the centre-surround diffuser state by *delta_time* seconds.

        Uses Euler time-stepping with a step size chosen to keep the IIR update
        coefficients (alpha values) below 1.  Stepping terminates early once the
        maximum absolute change per Euler step falls below
        :attr:`MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING`.

        Args:
            lp_log_frame: Current low-pass filtered log-intensity frame.
            cs_lambda_pixels: Space constant of the surround in pixels.
            delta_time: Elapsed time since the last frame in seconds.

        Returns:
            Updated surround frame array.
        """
        if self.cs_surround_frame is None:
            return lp_log_frame.copy()

        abs_min_tau_p = 1e-9
        tau_p = abs_min_tau_p if (self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) else self.cs_tau_p_ms * 1e-3
        tau_h = (
            abs_min_tau_p / (cs_lambda_pixels**2)
            if (self.cs_tau_h_ms is None or self.cs_tau_h_ms == 0)
            else self.cs_tau_h_ms * 1e-3
        )
        min_tau = min(tau_p, tau_h)
        NUM_STEPS_PER_TAU = 5
        num_steps = int(np.ceil((delta_time / min_tau) * NUM_STEPS_PER_TAU))
        actual_delta_time = delta_time / num_steps

        if num_steps > 1000 and not self.cs_steps_warning_printed:
            if self.cs_tau_p_ms == 0:
                logger.warning(
                    f"You set time constant cs_tau_p_ms to zero which set the minimum tau of {abs_min_tau_p}s"
                )
            logger.warning(
                f"CSDVS timestepping of diffuser could take up to {num_steps} "
                f"steps per frame for Euler delta time {actual_delta_time:.3g}s; "
                f"simulation of each frame will terminate when max change is smaller "
                f"than {EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING}"
            )
            self.cs_steps_warning_printed = True

        alpha_p = actual_delta_time / tau_p
        alpha_h = actual_delta_time / tau_h

        if alpha_p >= 1 or alpha_h >= 1:
            logger.error(
                f"CSDVS update alpha (of IIR update) is too large; simulation "
                f"would explode: alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}"
            )
            sys.exit(1)

        if alpha_p > 0.25 or alpha_h > 0.25:
            logger.warning(
                f"CSDVS update alpha (of IIR update) is too large; simulation "
                f"will be inaccurate: alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}"
            )

        surround: npt.NDArray = self.cs_surround_frame.astype(np.float32)
        max_change = 2 * EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING
        steps = 0

        while steps < num_steps and max_change > EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING:
            p_term = alpha_p * (lp_log_frame - surround)
            # mode='nearest' replicates border pixels, equivalent to ReplicationPad2d.
            h_conv: npt.NDArray = ndimage_convolve(surround, CSDVS_LAPLACIAN_KERNEL, mode="nearest")
            h_term = alpha_h * h_conv
            change = p_term + h_term
            max_change = float(np.max(np.abs(change)))
            surround = surround + change
            steps += 1

        return surround

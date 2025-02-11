"""
DVS simulator.
Compute events from input frames.
"""

import logging
import math
import random
import sys
from typing import Optional

import numpy as np
import torch  # https://pytorch.org/docs/stable/torch.html

from .emulator_utils import (
    compute_event_map,
    compute_photoreceptor_noise_voltage,
    generate_shot_noise,
    lin_log,
    low_pass_filter,
    rescale_intensity_frame,
    subtract_leak_current,
)

logger = logging.getLogger(__name__)


class EventEmulator(object):
    """compute events based on the input frame.
    - author: Tobi Delbruck, Yuhuang Hu, Zhe He
    - contact: tobi@ini.uzh.ch
    """

    MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING = 1e-5
    SCIDVS_GAIN: float = 2  # gain after highpass
    SCIDVS_TAU_S: float = 0.01  # small signal time constant in seconds
    SCIDVS_TAU_COV: float = (
        0.5  # each pixel has its own time constant. The tau's have log normal distribution with this sigma
    )

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
        seed: int = 0,
        cs_lambda_pixels: float = None,
        cs_tau_p_ms: float = None,
        hdr: bool = False,
        scidvs: bool = False,
        label_signal_noise=False,
    ):
        """
        Parameters
        ----------
        pos_thres: float, default 0.21
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.17
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float,
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: float
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz: float
            shot noise rate in Hz
        photoreceptor_noise: bool
            model photoreceptor noise to create the desired shot noise rate
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        cs_lambda_pixels: float
            space constant of surround in pixels, or None to disable surround inhibition
        cs_tau_p_ms: float
            time constant of lowpass filter of surround in ms or 0 to make surround 'instantaneous'
        hdr: bool
            Treat input as HDR floating point logarithmic gray scale with 255 input scaled as ln(255)=5.5441
        scidvs: bool
            Simulate the high gain adaptive photoreceptor SCIDVS pixel
        label_signal_noise: bool
            Record signal and noise event labels to a CSV file
        """

        self.no_events_warning_count = 0
        logger.info(
            "ON/OFF log_e temporal contrast thresholds: {} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres)
        )

        self.reset()
        self.t_previous = 0  # time of previous frame

        # thresholds
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres

        # non-idealities
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.photoreceptor_noise = photoreceptor_noise
        self.photoreceptor_noise_vrms: Optional[float] = None
        self.photoreceptor_noise_arr: Optional[np.ndarray] = (
            None  # separate noise source that is lowpass filtered to provide intensity-independent noise to add to intensity-dependent filtered photoreceptor output
        )
        if photoreceptor_noise:
            if shot_noise_rate_hz == 0:
                logger.warning(
                    "--photoreceptor_noise is specified but --shot_noise_rate_hz is 0; set a finite rate of shot noise events per pixel"
                )
                sys.exit(1)
            if cutoff_hz == 0:
                logger.warning(
                    "--photoreceptor_noise is specified but --cutoff_hz is zero; set a finite photoreceptor cutoff frequency"
                )
                sys.exit(1)
            self.photoreceptor_noise_samples = []

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.SHOT_NOISE_INTEN_FACTOR = 0.25  # this factor models the slight increase of shot noise with intensity

        # generate jax key for random process
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # event stats
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0

        # csdvs
        self.cs_steps_warning_printed = False
        self.cs_steps_taken = []
        self.cs_alpha_warning_printed = False
        self.cs_tau_p_ms = cs_tau_p_ms
        self.cs_lambda_pixels = cs_lambda_pixels
        self.cs_surround_frame: Optional[torch.Tensor] = None  # surround frame state
        self.csdvs_enabled = False  # flag to run center surround DVS emulation
        if self.cs_lambda_pixels is not None:
            self.csdvs_enabled = True
            # prepare kernels
            self.cs_tau_h_ms = (
                0
                if (self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0)
                else self.cs_tau_p_ms / (self.cs_lambda_pixels**2)
            )
            lat_res = 1 / (self.cs_lambda_pixels**2)
            trans_cond = 1 / self.cs_lambda_pixels
            logger.debug(
                f"lateral resistance R={lat_res:.2g}Ohm, transverse transconductance g={trans_cond:.2g} Siemens, Rg={(lat_res * trans_cond):.2f}"
            )
            self.cs_k_hh = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)
            # self.cs_k_pp = torch.tensor([[[[0, 0, 0],
            #                                [0, 1, 0],
            #                                [0, 0, 0]]]], dtype=torch.float32)
            logger.info(
                f"Center-surround parameters:\n\t"
                f"cs_tau_p_ms: {self.cs_tau_p_ms}\n\t"
                f"cs_tau_h_ms:  {self.cs_tau_h_ms}\n\t"
                f"cs_lambda_pixels:  {self.cs_lambda_pixels:.2f}\n\t"
            )

        # label signal and noise events
        self.label_signal_noise = label_signal_noise

        self.log_input = hdr
        if self.log_input:
            logger.info("Treating input as log-encoded HDR input")

        self.scidvs = scidvs
        if self.scidvs:
            logger.info("Modeling potential SCIDVS pixel with nonlinear CR highpass amplified log intensity")

    def _init(self, first_frame_linear):
        """

        Parameters:
        ----------
        first_frame_linear: np.ndarray
            the first frame, used to initialize data structures

        Returns:
            new instance
        -------

        """
        logger.debug("initializing random temporal contrast thresholds from from base frame")
        # base_frame are memorized lin_log pixel values
        self.diff_frame = None

        # take the variance of threshold into account.
        if self.sigma_thres > 0:
            self.pos_thres = torch.normal(
                self.pos_thres, self.sigma_thres, size=first_frame_linear.shape, dtype=torch.float32
            )

            # to avoid the situation where the threshold is too small.
            self.pos_thres = torch.clamp(self.pos_thres, min=0.01)

            self.neg_thres = torch.normal(
                self.neg_thres, self.sigma_thres, size=first_frame_linear.shape, dtype=torch.float32
            )
            self.neg_thres = torch.clamp(self.neg_thres, min=0.01)

        # compute variable for shot-noise
        self.pos_thres_pre_prob = torch.div(self.pos_thres_nominal, self.pos_thres)
        self.neg_thres_pre_prob = torch.div(self.neg_thres_nominal, self.neg_thres)

        if self.scidvs and EventEmulator.SCIDVS_TAU_COV > 0:
            self.scidvs_tau_arr = EventEmulator.SCIDVS_TAU_S * (
                torch.exp(
                    torch.normal(0, EventEmulator.SCIDVS_TAU_COV, size=first_frame_linear.shape, dtype=torch.float32)
                )
            )

        # If leak is non-zero, then initialize each pixel memorized value
        # some fraction of ON threshold below first frame value, to create leak
        # events from the start; otherwise leak would only gradually
        # grow over time as pixels spike.
        # do this *AFTER* we determine randomly distributed thresholds
        # (and use the actual pixel thresholds)
        # otherwise low threshold pixels will generate
        # a burst of events at the first frame
        if self.leak_rate_hz > 0:
            # no justification for this subtraction after having the
            # new leak rate model
            #  self.base_log_frame -= torch.rand(
            #      first_frame_linear.shape,
            #      dtype=torch.float32)*self.pos_thres

            # set noise rate array, it's a log-normal distribution
            self.noise_rate_array = torch.randn(first_frame_linear.shape, dtype=torch.float32)
            self.noise_rate_array = torch.exp(math.log(10) * self.noise_rate_cov_decades * self.noise_rate_array)

        # refractory period
        if self.refractory_period_s > 0:
            self.timestamp_mem = torch.zeros(first_frame_linear.shape, dtype=torch.float32) - self.refractory_period_s

        # scidvs adaptation

    def scidvs_dvdt(self, v, tau=None):
        """

        Parameters
        ----------
            the input 'voltage',
        v:Tensor
            actually log intensity in base e units
        tau:Optional[Tensor]
            if None, tau is set internally

        Returns
        -------
        the time derivative of the signal

        """
        if tau is None:
            tau = EventEmulator.SCIDVS_TAU_S  # time constant for small signals = C/g
        # C = 100e-15
        # g = C/tau
        efold = 1 / 0.7  # efold of sinh conductance in log_e units, based on 1/kappa
        dvdt = torch.div(1, tau) * torch.sinh(v / efold)
        return dvdt

    def set_dvs_params(self, model: str):
        if model == "clean":
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.02
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.leak_jitter_fraction = 0
            self.noise_rate_cov_decades = 0
            self.shot_noise_rate_hz = 0  # rate in hz of temporal noise events
            self.refractory_period_s = 0

        elif model == "noisy":
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.05
            self.cutoff_hz = 30
            self.leak_rate_hz = 0.1
            # rate in hz of temporal noise events
            self.shot_noise_rate_hz = 5.0
            self.refractory_period_s = 0
            self.leak_jitter_fraction = 0.1
            self.noise_rate_cov_decades = 0.1
        else:
            #  logger.error(
            #      "dvs_params {} not known: "
            #      "use 'clean' or 'noisy'".format(model))
            logger.warning("dvs_params {} not known: Using commandline assigned options".format(model))
            #  sys.exit(1)
        logger.info(
            "set DVS model params with option '{}' "
            "to following values:\n"
            "pos_thres={}\n"
            "neg_thres={}\n"
            "sigma_thres={}\n"
            "cutoff_hz={}\n"
            "leak_rate_hz={}\n"
            "shot_noise_rate_hz={}\n"
            "refractory_period_s={}".format(
                model,
                self.pos_thres,
                self.neg_thres,
                self.sigma_thres,
                self.cutoff_hz,
                self.leak_rate_hz,
                self.shot_noise_rate_hz,
                self.refractory_period_s,
            )
        )

    def reset(self):
        """resets so that next use will reinitialize the base frame"""
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0

        self.new_frame: Optional[np.ndarray] = None  # new frame that comes in [height, width]
        self.log_new_frame: Optional[np.ndarray] = None  #  [height, width]
        self.lp_log_frame: Optional[np.ndarray] = None  # lowpass stage 0
        self.lp_log_frame: Optional[np.ndarray] = None  # stage 1
        self.cs_surround_frame: Optional[np.ndarray] = None
        self.c_minus_s_frame: Optional[np.ndarray] = None
        self.base_log_frame: Optional[np.ndarray] = None  # memorized log intensities at change detector
        self.diff_frame: Optional[np.ndarray] = None  # [height, width]
        self.scidvs_highpass: Optional[np.ndarray] = None
        self.scidvs_previous_photo: Optional[np.ndarray] = None
        self.scidvs_tau_arr: Optional[np.ndarray] = None

        self.frame_counter = 0

    def generate_events(self, new_frame, t_frame):
        """Compute events in new frame.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width], NOTE y is first dimension, like in matlab the column, x is 2nd dimension, i.e. row.
        t_frame: float
            timestamp of new frame in float seconds

        Returns
        -------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
            NOTE x,y, NOT y,x.
        """

        # base_frame: the change detector input,
        #              stores memorized brightness values
        # new_frame: the new intensity frame input
        # log_frame: the lowpass filtered brightness values

        # update frame counter
        self.frame_counter += 1

        if t_frame < self.t_previous:
            raise ValueError(
                "this frame time={} must be later than previous frame time={}".format(t_frame, self.t_previous)
            )

        # compute time difference between this and the previous frame
        delta_time = t_frame - self.t_previous
        # logger.debug('delta_time={}'.format(delta_time))

        if self.log_input and new_frame.dtype != np.float32:
            logger.warning("log_frame is True but input frome is not np.float32 datatype")

        # convert into torch tensor
        self.new_frame = torch.tensor(new_frame, dtype=torch.float64)
        # lin-log mapping, if input is not already float32 log input
        self.log_new_frame = lin_log(self.new_frame) if not self.log_input else self.new_frame

        inten01 = None  # define for later
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # Time constant of the filter is proportional to
            # the intensity value (with offset to deal with DN=0)
            # limit max time constant to ~1/10 of white intensity level
            inten01 = rescale_intensity_frame(self.new_frame.clone().detach())  # TODO assumes 8 bit

        # Apply nonlinear lowpass filter here.
        # Filter is a 1st order lowpass IIR (can be 2nd order)
        # that uses two internal state variables
        # to store stages of cascaded first order RC filters.
        # Time constant of the filter is proportional to
        # the intensity value (with offset to deal with DN=0)
        if self.base_log_frame is None:
            # initialize 1st order IIR to first input
            self.lp_log_frame = self.log_new_frame
            self.photoreceptor_noise_arr = torch.zeros_like(self.lp_log_frame)

        self.lp_log_frame = low_pass_filter(
            log_new_frame=self.log_new_frame,
            lp_log_frame=self.lp_log_frame,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cutoff_hz,
        )

        # add photoreceptor noise if we are using photoreceptor noise to create shot noise
        if (
            self.photoreceptor_noise and self.base_log_frame is not None
        ):  # only add noise after the initial values are memorized and we can properly lowpass filter the noise
            self.photoreceptor_noise_vrms = compute_photoreceptor_noise_voltage(
                shot_noise_rate_hz=self.shot_noise_rate_hz,
                f3db=self.cutoff_hz,
                sample_rate_hz=1 / delta_time,
                pos_thr=self.pos_thres_nominal,
                neg_thr=self.neg_thres_nominal,
                sigma_thr=self.sigma_thres,
            )
            noise = self.photoreceptor_noise_vrms * torch.randn(self.log_new_frame.shape, dtype=torch.float32)
            self.photoreceptor_noise_arr = low_pass_filter(
                noise, self.photoreceptor_noise_arr, None, delta_time, self.cutoff_hz
            )
            self.photoreceptor_noise_samples.append(
                self.photoreceptor_noise_arr[0, 0].cpu().item()
            )  # todo debugging can remove
            # std=np.std(self.photoreceptor_noise_samples)

        # surround computations by time stepping the diffuser
        if self.csdvs_enabled:
            self._update_csdvs(delta_time)

        if self.base_log_frame is None:
            self._init(new_frame)
            if not self.csdvs_enabled:
                self.base_log_frame = self.lp_log_frame
            else:
                self.base_log_frame = (
                    self.lp_log_frame - self.cs_surround_frame
                )  # init base log frame (input to diff) to DC value, TODO check might not be correct to avoid transient

            return None  # on first input frame we just setup the state of all internal nodes of pixels

        if self.scidvs:
            if self.scidvs_highpass is None:
                self.scidvs_highpass = torch.zeros_like(self.lp_log_frame)
                self.scidvs_previous_photo = torch.clone(self.lp_log_frame).detach()
            self.scidvs_highpass += (self.lp_log_frame - self.scidvs_previous_photo) - delta_time * self.scidvs_dvdt(
                self.scidvs_highpass, self.scidvs_tau_arr
            )
            self.scidvs_previous_photo = torch.clone(self.lp_log_frame)

        # Leak events: switch in diff change amp leaks at some rate
        # equivalent to some hz of ON events.
        # Actual leak rate depends on threshold for each pixel.
        # We want nominal rate leak_rate_Hz, so
        # R_l=(dI/dt)/Theta_on, so
        # R_l*Theta_on=dI/dt, so
        # dI=R_l*Theta_on*dt
        if self.leak_rate_hz > 0:
            self.base_log_frame = subtract_leak_current(
                base_log_frame=self.base_log_frame,
                leak_rate_hz=self.leak_rate_hz,
                delta_time=delta_time,
                pos_thres=self.pos_thres,
                leak_jitter_fraction=self.leak_jitter_fraction,
                noise_rate_array=self.noise_rate_array,
            )

        # log intensity (brightness) change from memorized values is computed
        # from the difference between new input
        # (from lowpass of lin-log input) and the memorized value

        # take input from either photoreceptor or amplified high pass nonlinear filtered scidvs
        photoreceptor = EventEmulator.SCIDVS_GAIN * self.scidvs_highpass if self.scidvs else self.lp_log_frame

        if not self.csdvs_enabled:
            self.diff_frame = photoreceptor + self.photoreceptor_noise_arr - self.base_log_frame
        else:
            self.c_minus_s_frame = photoreceptor + self.photoreceptor_noise_arr - self.cs_surround_frame
            self.diff_frame = self.c_minus_s_frame - self.base_log_frame

        # generate event map
        # print(f'\ndiff_frame max={torch.max(self.diff_frame)} pos_thres mean={torch.mean(self.pos_thres)} expect {int(torch.max(self.diff_frame)/torch.mean(self.pos_thres))} max events')
        pos_evts_frame, neg_evts_frame = compute_event_map(self.diff_frame, self.pos_thres, self.neg_thres)
        max_num_events_any_pixel = max(
            pos_evts_frame.max(), neg_evts_frame.max()
        )  # max number of events in any pixel for this interframe
        max_num_events_any_pixel = max_num_events_any_pixel.cpu().numpy().item()  # turn singleton tensor to scalar
        if max_num_events_any_pixel > 100:
            logger.warning(f"Too many events generated for this frame: num_iter={max_num_events_any_pixel}>100 events")

        # to assemble all events
        events = torch.empty(
            (0, 4), dtype=torch.float32
        )  # ndarray shape (N,4) where N is the number of events are rows are [t,x,y,p]
        # event timestamps at each iteration
        # min_ts_steps timestamps are linearly spaced
        # they start after the self.t_previous to make sure
        # that there is interval from previous frame
        # they end at t_frame.
        # delta_time=t_frame - self.t_previous
        # e.g. t_start=0, t_end=1, min_ts_steps=2, i=0,1
        # ts=1*1/2, 2*1/2
        #  ts = self.t_previous + delta_time * (i + 1) / min_ts_steps
        # if min_ts_steps==1, then there is only a single timestamp at t_frame
        min_ts_steps = max_num_events_any_pixel if max_num_events_any_pixel > 0 else 1
        ts_step = delta_time / min_ts_steps
        ts = torch.linspace(start=self.t_previous + ts_step, end=t_frame, steps=min_ts_steps, dtype=torch.float32)
        # print(f'ts={ts}')

        # record final events update
        final_pos_evts_frame = torch.zeros(pos_evts_frame.shape, dtype=torch.int32)
        final_neg_evts_frame = torch.zeros(neg_evts_frame.shape, dtype=torch.int32)

        if max_num_events_any_pixel == 0 and self.no_events_warning_count < 100:
            logger.warning(f"no signal events generated for frame #{self.frame_counter:,} at t={t_frame:.4f}s")
            self.no_events_warning_count += 1
            # max_num_events_any_pixel = 1
        else:  # there are signal events to generate
            for i in range(max_num_events_any_pixel):
                # events for this iteration

                # already have the number of events for each pixel in
                # pos_evts_frame, just find bool array of pixels with events in
                # this iteration of max # events

                # it must be >= because we need to make event for
                # each iteration up to total # events for that pixel
                pos_cord = pos_evts_frame >= i + 1
                neg_cord = neg_evts_frame >= i + 1

                # filter events with refractory_period
                # only filter when refractory_period_s is large enough
                # otherwise, pass everything
                # TODO David Howe figured out that the reference level was resetting to the log photoreceptor value at event generation,
                # NOT at the value at the end of the refractory period.
                # Brian McReynolds thinks that this effect probably only makes a significant difference if the temporal resolution of the signal
                # is high enough so that dt is less than one refractory period.
                if self.refractory_period_s > ts_step:
                    pos_time_since_last_spike = pos_cord * ts[i] - self.timestamp_mem
                    neg_time_since_last_spike = neg_cord * ts[i] - self.timestamp_mem

                    # filter the events
                    pos_cord = pos_time_since_last_spike > self.refractory_period_s
                    neg_cord = neg_time_since_last_spike > self.refractory_period_s

                    # assign new history
                    self.timestamp_mem = torch.where(pos_cord, ts[i], self.timestamp_mem)
                    self.timestamp_mem = torch.where(neg_cord, ts[i], self.timestamp_mem)

                # update event count frames with the shot noise
                final_pos_evts_frame += pos_cord
                final_neg_evts_frame += neg_cord

                # generate events
                # make a list of coordinates x,y addresses of events
                # torch.nonzero(as_tuple=True)
                # Returns a tuple of 1-D tensors, one for each dimension in input,
                # each containing the indices (in that dimension) of all non-zero elements of input .

                # pos_event_xy and neg_event_xy each return two 1-d tensors each with same length of the number of events
                #   Tensor 0 is list of y addresses (first dimension in pos_cord input)
                #   Tensor 1 is list of x addresses
                pos_event_xy = pos_cord.nonzero(as_tuple=True)
                neg_event_xy = neg_cord.nonzero(as_tuple=True)

                events_curr_iter = self.get_event_list_from_coords(pos_event_xy, neg_event_xy, ts[i])

                # shuffle and append to the events collectors
                if events_curr_iter is not None:
                    idx = torch.randperm(events_curr_iter.shape[0])
                    events_curr_iter = events_curr_iter[idx].view(events_curr_iter.size())
                    events = torch.cat((events, events_curr_iter))

                # end of iteration over max_num_events_any_pixel

        # NOISE: add shot temporal noise here by
        # simple Poisson process that has a base noise rate
        # self.shot_noise_rate_hz.
        # If there is such noise event,
        # then we output event from each such pixel. Note this is too simplified to model
        # alternating ON/OFF noise; see --photoreceptor_noise option for that type of noise
        # Advantage here is to be able to label signal and noise events.

        # the shot noise rate varies with intensity:
        # for lowest intensity the rate rises to parameter.
        # the noise is reduced by factor
        # SHOT_NOISE_INTEN_FACTOR for brightest intensities

        shot_on_cord, shot_off_cord = None, None

        num_signal_events = len(events)
        signnoise_label = (
            torch.ones(num_signal_events, dtype=torch.bool) if self.label_signal_noise else None
        )  # all signal so far

        # This was in the loop, here we calculate loop-independent quantities
        if self.shot_noise_rate_hz > 0 and not self.photoreceptor_noise:
            # generate all the noise events for this entire input frame; there could be (but unlikely) several per pixel but only 1 on or off event is returned here
            shot_on_cord, shot_off_cord = generate_shot_noise(
                shot_noise_rate_hz=self.shot_noise_rate_hz,
                delta_time=delta_time,
                shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
                inten01=inten01,
                pos_thres_pre_prob=self.pos_thres_pre_prob,
                neg_thres_pre_prob=self.neg_thres_pre_prob,
            )

            # noise_on_xy and noise_off_xy each are two 1-d tensors each with same length of the number of events
            #   Tensor 0 is list of y addresses (first dimension in pos_cord input)
            #   Tensor 1 is list of x addresses
            shot_on_xy = shot_on_cord.nonzero(as_tuple=True)
            shot_off_xy = shot_off_cord.nonzero(as_tuple=True)

            # give noise events the last timestamp generated for any signal event from this frame
            shot_noise_events = self.get_event_list_from_coords(shot_on_xy, shot_off_xy, ts[-1])

            # append the shot noise events and shuffle in, keeping track of labels if labeling
            # append to the signal events but don't shuffle since this causes nonmonotonic timestamps
            if shot_noise_events is not None:
                num_shot_noise_events = len(shot_noise_events)
                events = torch.cat((events, shot_noise_events), dim=0)  # stack signal events before noise events, [N,4]
                # idx = torch.randperm(num_total_events)  # makes timestamps nonmonotonic
                # events = events[idx].view(events.size())
                if self.label_signal_noise:
                    noise_label = torch.zeros((num_shot_noise_events), dtype=torch.bool)
                    signnoise_label = torch.cat((signnoise_label, noise_label))
                    signnoise_label = signnoise_label[idx].view(signnoise_label.size())

        # update base log frame according to the final
        # number of output events
        # update the base frame, after we know how many events per pixel
        # add to memorized brightness values just the events we emitted.
        # don't add the remainder.
        # the next aps frame might have sufficient value to trigger
        # another event, or it might not, but we are correct in not storing
        # the current frame brightness
        #  self.base_log_frame += pos_evts_frame*self.pos_thres
        #  self.base_log_frame -= neg_evts_frame*self.neg_thres

        self.base_log_frame += (
            final_pos_evts_frame * self.pos_thres
        )  # TODO should this be self.lp_log_frame ? I.e. output of lowpass photoreceptor?
        self.base_log_frame -= final_neg_evts_frame * self.neg_thres

        # however, if we made a shot noise event, then just memorize the log intensity at this point, so that the pixels are reset and forget the log intensity input
        if not self.photoreceptor_noise and self.shot_noise_rate_hz > 0:
            self.base_log_frame[shot_on_xy] = self.lp_log_frame[shot_on_xy]
            self.base_log_frame[shot_off_xy] = self.lp_log_frame[shot_off_xy]

        if len(events) > 0:
            events = (
                events.cpu().data.numpy()
            )  # # ndarray shape (N,4) where N is the number of events are rows are [t,x,y,p]
            timestamps = events[:, 0]
            if np.any(np.diff(timestamps) < 0):
                idx = np.argwhere(np.diff(timestamps) < 0)
                logger.warning(f"nonmonotonic timestamp(s) at indices {idx}")
            if signnoise_label is not None:
                signnoise_label = signnoise_label.cpu().numpy()

        # assign new time
        self.t_previous = t_frame
        if len(events) > 0:
            # debug TODO remove
            tsout = events[:, 0]
            tsoutdiff = np.diff(tsout)
            if np.any(tsoutdiff < 0):
                print("nonmonotonic timestamp in events")

            return events  # ndarray shape (N,4) where N is the number of events are rows are [t,x,y,p]. Confirmed by Tobi Oct 2023
        else:
            return None

    def get_event_list_from_coords(self, pos_event_xy, neg_event_xy, ts):
        """Gets event list from ON and OFF event coordinate lists.
        :param pos_event_xy: Tensor[2,n] where n is number of ON events, [0,n] are y addresses and [1,n] are x addresses
        :param neg_event_xy: Tensor[2,m] where m is number of ON events, [0,m] are y addresses and [1,m] are x addresses
        :param ts: the timestamp given to all events (scalar)
        :returns: Tensor[n+m,4] of AER [t, x, y, p]
        """
        # update event stats
        num_pos_events = pos_event_xy[0].shape[0]
        num_neg_events = neg_event_xy[0].shape[0]
        num_events = num_pos_events + num_neg_events
        events_curr_iter = None
        if num_events > 0:
            # following will update stats for all events (signal and shot noise)
            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            # events_curr_iter is 2d array [N,4] with 2nd dimension [t,x,y,p]
            events_curr_iter = torch.ones(  # set all elements 1 so that polarities start out positive ON events
                (num_events, 4), dtype=torch.float32
            )
            events_curr_iter[:, 0] *= ts  # put all timestamps into events

            # pos_event cords
            # events_curr_iter is 2d array [N,4] with 2nd dimension [t,x,y,p]. N is the number of events from this frame
            # we replace the x's (element 1) and y's (element 2) with the on event coordinates in the first num_pos_coord entries of events_curr_iter
            events_curr_iter[:num_pos_events, 1] = pos_event_xy[1]  # tensor 1 of pos_event_xy is x addresses
            events_curr_iter[:num_pos_events, 2] = pos_event_xy[0]  # tensor 0 of pos_event_xy is y addresses

            # neg event cords
            # we replace the x's (element 1) and y's (element 2) with the off event coordinates in the remaining entries num_pos_events: entries of events_curr_iter
            events_curr_iter[num_pos_events:, 1] = neg_event_xy[1]
            events_curr_iter[num_pos_events:, 2] = neg_event_xy[0]
            events_curr_iter[num_pos_events:, 3] = -1  # neg events polarity is -1 so flip the signs
        return events_curr_iter

    def _update_csdvs(self, delta_time):
        if self.cs_surround_frame is None:
            self.cs_surround_frame = (
                self.lp_log_frame.clone().detach()
            )  # detach makes true clone decoupled from torch computation tree
        else:
            # we still need to simulate dynamics even if "instantaneous", unfortunately it will be really slow with Euler stepping and
            # no gear-shifting
            # TODO change to compute steady-state 'instantaneous' solution by better method than Euler stepping
            abs_min_tau_p = 1e-9
            tau_p = abs_min_tau_p if (self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) else self.cs_tau_p_ms * 1e-3
            tau_h = (
                abs_min_tau_p / (self.cs_lambda_pixels**2)
                if (self.cs_tau_h_ms is None or self.cs_tau_h_ms == 0)
                else self.cs_tau_h_ms * 1e-3
            )
            min_tau = min(tau_p, tau_h)
            # if min_tau < abs_min_tau_p:
            #     min_tau = abs_min_tau_p
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
                    f"simulation of each frame will terminate when max change is smaller than {EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING}"
                )
                self.cs_steps_warning_printed = True

            alpha_p = actual_delta_time / tau_p
            alpha_h = actual_delta_time / tau_h
            if alpha_p >= 1 or alpha_h >= 1:
                logger.error(
                    f"CSDVS update alpha (of IIR update) is too large; simulation would explode: "
                    f"alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}"
                )
                self.cs_alpha_warning_printed = True
                sys.exit(1)
            if alpha_p > 0.25 or alpha_h > 0.25:
                logger.warning(
                    f"CSDVS update alpha (of IIR update) is too large; simulation will be inaccurate: "
                    f"alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}"
                )
                self.cs_alpha_warning_printed = True
            p_ten = torch.unsqueeze(torch.unsqueeze(self.lp_log_frame, 0), 0)
            h_ten = torch.unsqueeze(torch.unsqueeze(self.cs_surround_frame, 0), 0)
            padding = torch.nn.ReplicationPad2d(1)
            max_change = 2 * EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING
            steps = 0
            while steps < num_steps and max_change > EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING:
                diff = p_ten - h_ten
                p_term = alpha_p * diff
                # For the conv2d, unfortunately the zero padding pulls down the border pixels,
                # so we use replication padding to reduce this effect on border.
                # TODO check if possible to implement some form of open circuit resistor termination condition by correct padding
                h_conv = torch.conv2d(padding(h_ten.float()), self.cs_k_hh.float())
                h_term = alpha_h * h_conv
                change_ten = p_term + h_term  # change_ten is the change in the diffuser voltage
                max_change = torch.max(
                    torch.abs(change_ten)
                ).item()  # find the maximum absolute change in any diffuser pixel
                h_ten += change_ten
                steps += 1

            self.cs_steps_taken.append(steps)
            self.cs_surround_frame = torch.squeeze(h_ten)

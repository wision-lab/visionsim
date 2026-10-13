"""iToF (indirect time-of-flight) sensor emulation."""

from .coding import (  # noqa: F401
    CodingScheme,
    make_coding_functions,
    make_conv_sinusoidal_codes,
    make_conv_square_codes,
    make_delta_hilbert_codes,
    make_delta_sinusoidal_codes,
    make_double_ramp_codes,
    make_gray_codes,
    make_gray_codes_reduced,
    make_max_min_run_length_gray_codes,
    make_multi_freq_sinusoidal_codes,
    make_single_ramp_codes,
    make_tof_gray_codes,
    make_tof_hilbert_codes,
    unambiguous_range,
)
from .decoding import (  # noqa: F401
    decode,
    decode_double_ramp,
    decode_hilbert,
    decode_mult_freq_sinusoid,
    decode_single_ramp,
    decode_sinusoid,
    decode_square,
)
from .simulation import (  # noqa: F401
    compute_correlation_function,
    simulate_measurements,
)

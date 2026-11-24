"""dToF (direct time-of-flight) sensor emulation."""

from .histogrammers import (  # noqa: F401
    forward_free_running,
    forward_sync,
    simulate_ewh,
    simulate_photon_timestamps,
    timestamps_to_histogram,
)
from .simulation import (  # noqa: F401
    DtoFModel,
    apply_irf,
    calculate_transients,
    emulate_dtof,
    make_irf,
)

Roadmap
=======

This roadmap is loosely structured in order of priority. Some items are mostly implemented and need
some additional careful testing before release, while others are planned and might not materialize.
It does not constitute a promise of future features; instead it serves as an outlook of what is to
come. Much work and research is yet to come, and as always, feel free to
`contribute <https://github.com/WISION-Lab/visionsim/blob/main/CONTRIBUTING.md>`_.

- Add direct time-of-flight single photon camera (e.g. LiDAR) support.
- Add a conventional camera ISP model (e.g. demosaicing, color correction, tone mapping).
- Add thermal camera modeling.
- Add dynamic thermal simulation; only steady state has shipped so far.
- Allow iToF/dToF to use non-uniform background illumination.
- Add noise modeling to iToF emulation (photon shot noise and read noise).
- Model angular acceleration in IMU emulation, which currently ignores it.
- Support interpolation methods beyond RIFE and factors that are not powers of two.

Long-term Focus
---------------

- Homogenize the CLI so that common argument names are shared and behave as expected
  (e.g. ``--force``).
- Standardize the sensor emulation API.
- Improve sensor emulation speed.
- Add other sensor modalities, such as radar and structured light.
- Add better integration with other tools and simulators (e.g. ROS, CARLA, Unreal Engine,
  Genesis).

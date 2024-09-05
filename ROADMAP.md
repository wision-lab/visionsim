# Proposal

Single photon cameras (SPCs) are an emerging class of sensors that offer extreme sensitivity and are manufactured at scale using existing CMOS techniques. These sensors can not only detect individual photons but also time them with picosecond accuracy, enabling applications that are [not possible with traditional sensors](https://wisionlab.com/project/burst-vision-single-photon/). By sensing individual photons, SPCs provide us with the most fine-grained information possible and provides a lot of flexibility to the authors of downstream inference tasks.

However, today there exists no easy way to simulate SPCs under a wide range of settings efficiently and at scale, which severely impedes rapid development of new techniques and limiting their adoption. This work aims to provide such tools to the community in order to facilitate the development of new SPC-based techniques and provide a standardized way to do so.

# High Level Architecture

The single photon simulation engine today is made up of four layers and accessible as both a CLI and library. These layers are as follows:

- Ground Truth Simulation Layer: Generate ground truth data using existing high-quality rendering engines via either custom plugins or render scripts. The output of this layer will consist of clean, blur and noise free, RGB images, depth maps, as well as segmentation maps, normal maps, and other data modalities where applicable. These should be stored in a common format described below, or something that the next later can ingest.

- Interpolation Layer: The simulated data from layer#1, or in fact, data from existing datasets such as the [XVFI](https://github.com/JihyongOh/XVFI), can be interpolated to yield higher framerate datasets. The reasoning for this layer is two fold: i) Simulations can be slow and expensive to run, especially when the single photon sensors run at hundreds of thousands of frames per second, filling the gaps between adjacent frames is fast and relatively cheap to compute. Similarly, existing datasets can be uplifted to single photon regimes using interpolation techniques. ii) While interpolation can produce some artifacts, these are minor when the frames we interpolate between are close and are further muddled as we use the interpolated frames to emulate different camera modalities.

- Emulation Layer: At this stage we have access to high speed interpolated data which we'll need to further process into the desired modality. We'll refer to this process as `emulation` and reserve the term `simulation` for layer#1. Specifically we are interested in the following:
  
  - Passive Single Photon Cameras: These can be emulated by Bernoulli sampling the interpolated intensity frames.
  - Active Single Photon Cameras: By defining a few added parameters such as laser power, repetition rate, wavelength, etc, we can use the interpolated depth, normals and intensity to construct arrays of histograms with preset (non negligible) field-of-views that will have realistic blur, depth edges, etc.
  - Conventional RGB Cameras: Many interpolated intensity frames can be averaged together to create motion blur. We then add Gaussian read noise and quantize to a certain bit-depth. More complex effects such as tonemapping, optical aberrations, etc, can be performed here as well.
  - Potentially others to come!

- Data Storage Layer: Finally, all this data must be stored alongside all applicable metadata in a way that enables easy iteration and random access which is crucial for any deep learning applications. This data format is further detailed below.

## Folder Structure, Data Format, and Schema

Datasets of a single scene will have the following structure:

```
SCENE-NAME
├── renders
│   ├── train
│   │   ├── frames.npy
│   │   └── transforms.json
│   ├── test/
│   └── val/
├── interpolated/
├── binary
│   ├── train
│   │   ├── frames.npy
│   │   └── transforms.json
│   ├── test/
│   └── val/
└── rgb
    ├── train
    │   ├── frames/
    │   │   ├── frame_000000.png
    │   │   ├── frame_000001.png
    │   │   └── ...
    │   ├── preview.mp4
    │   └── transforms.json
    ├── test/
    └── val/
```

*Note:* Here we are only showing the training sets for brevity. Testing and validations sets will be similar in layout.

Here there are four datasets, the simulated ground truth dataset (`renders/`), it's interpolated counterpart (`interpolated/`) emulated SPC dataset (`binary/`), emulated RGB (`rgb/`). While the interpolated dataset is needed to create the last two, it is usually not needed and hence deleted after the other datasets have been created.

A _full_ dataset refers to a folder containing a train/test/val folder, it is uniquely identified by the path of the parent folder, for example `SCENE-NAME/binary`. More loosely speaking, a (not-full) dataset might not contain different subsets and is identified by it's path (i.e: `SCENE-NAME/rgb/train`) or the path of it's transform file directly.

Every dataset, or subset thereof, needs to contain a valid `transforms.json` file which contains camera parameters, extrinsics, and among other things, the paths to the data. This format is meant to be a superset of older NERF-like datasets and should be familiar to a lot of users. Currently they come in two formats, the `NPY` format and the `IMG` format. CLI tools are provided to convert between the two.

### The `IMG` Format

This format can be see in the above example in the `rgb` dataset. Here, the raw data is simply stored as a collection of images in `frames/`. The transforms file adheres to, at a minimum, the following schema:

```
IMG_SCHEMA = {
    "fl_x": Focal length in X
    "fl_y": Focal length in Y
    "cx": Center of optical axis in pixel coordinates in X
    "cy": Center of optical axis in pixel coordinates in Y
    "w": Sensor width in pixels
    "h": Sensor height in pixels
    "c": Number of output channels (i.e: RGBA = 4)
    "frames": List of frames where each frame follows IMG_FRAME_SCHEMA
}

IMG_FRAME_SCHEMA = {
    "transform_matrix": 4x4 transform matrix
    "file_path": Path to color image, relative to parent dir
}
```

### The `NPY` Format

Here, we instead store the data as a `.npy` file which has several benefits:

- Numpy arrays in this format can be memory mapped, enabling us to manipulate larger-than-RAM arrays efficiently and easily. This is particularly important when sampling pixels as we do not need to read and decode a whole image to sample a single pixel.

- We are not constrained by an image codec, meaning we can for instance use higher precision floats (although EXRs support this), losslessly save data, and in the case of binary valued SPC data, we can bitpack it leading to larger-than-PNG compression levels.

- Dataset integrity is easier to check as it's just an npy file and it's accompanying transforms file as opposed to thousands of image files.

...and a few drawbacks:

- Not directly compatible with existing tooling, although reading from an npy file is easy.

- Can create very large single files.

The schema is very similar except now there's a single `file_path`, as well as a few additional keys detailing if the data is bitpacked:

```
NPY_SCHEMA = {
    "fl_x", "fl_y", "cx", "cy", "h", "w", "c": Same as above
    "bitpack": Whether npy file is bitpacked
    "bitpack_dim": Bitpacked axis
    "file_path": Path to npy file containing color images
    "frames": List of frames where each frame follows NPY_FRAME_SCHEMA
}

NPY_FRAME_SCHEMA = {
    "transform_matrix": 4x4 transform matrix
}
```

<!-- ## Physical Units -->

<!-- ## Use Cases and Granularity of API -->

# Roadmap

## Phase 1: Polish passive SpSIM, release v0.1.0 publicly

This release, with target release date of late 2024, will focus solely on passive SPC simulation using blender. The goal is to release a minimum-viable-simulator, that is reliable and slimmed down in scope, as soon as possible. For this to happen we need to do the following:

- [ ] Reset the version numbers to 0.0.1 and adhere to strict [semver](https://semver.org/) versioning going forward with the goal of a 0.1.0 public release. The version can safely be rolled back as this project is not public yet.

- [ ] Update and polish the documentation. Some docs were written by @mischemer, these need to be expanded to fully document both the CLI and public API as well as having a quickstart guide and basic tutorial.
  
  - [ ] Add per task examples and documentation.

- [ ] Streamline installation process. Perhaps with a `post-install` task?

- [ ] Setup a rigorous testing suite, if possible that tests different versions of blender too (maybe using docker).

- [ ] Revise and refactor CLI and library API such that their functionality is on par with each other.

- [ ] Improve logging (maybe with rich so it can work with the renderer)

- [ ] Fix or remove TNB frame from blender.render. Slim down CLI as needed.

- [ ] Remove depth/normal simulation capabilities as a change of schema will be needed to fully accommodate them.

- [ ] Deal with alpha channel in a consistent manner. Currently different CLI tools approach this issue in different manners.
  
  - [ ] If background color is set in blender render, output RGB instead of RGBA?

- [ ] Consolidate checks (ffmpeg -V, etc) to only run once per spsim invocation.

- [ ] Add debug parameter globally, maybe as environment variable which would dry run/echo commands. There's already a max threads environment variable, which also needs to be documented. 

- [ ] Remove dependency on .io and .utils

## Phase 2: Enable depth and normal generation, Provide higher-level API, release v0.2.0

This phase has two primary goals:

- [ ] (re)Enable depth and normal generation: While it's easy to generate these via blender, the dataset schema and all CLI and library code needs to be updated to properly account for these new modalities.

- [ ] Provide higher-level API for common tasks: Phase 1 provided the basic tools needed to create datasets but requires a lot of boilerplate to do so and to get the correct folder structure. A new task could be introduced to create a full dataset more easily. Importantly, how do we do so in a way that's forward compatible and does not lead to CLI argument bloat?

## Phase 3: Add transient emulation layer, CARLA simulation engine plugin, release v0.3.0

With depth and normals, we can now emulate transients given a few additional sensor parameters. Modeling transient histogram formation correctly is tricky, there's a lot of physical processes to take into account, and simulating all of them might not make sense either. A lot of the ground work for simulating transients in this manner can be found in the [flamingo simulator](https://github.com/jungerm2/flamingo) repository. It also contains code to interface with the [CARLA](https://carla.org/) simulation engine, which is more generally geared towards self driving vehicles. 

Deliverables:

- [ ] Add transient emulation layer taken from flamingo.
  
  - [ ] Come up with a simple way to specify transient field-of-view, array configuration, bin width and count, light sources (active and ambient), and any other required added information. We might need to supplement the transforms file with a dedicated file describing these settings, or update the schema in a backwards compatible way.
  
  - [ ] Describe the transient emulation process, what is simulated and what is explicitly skipped and add this to the documentation. 

- [ ] Integrate CARLA into SpSIM, i.e: cannibalize flamingo's CLI and create a `carla.render` task.
  
  - [ ] Focus on minimum CARLA release first,  and add in customization slowly afterwards. 
  
  - [ ] Newer versions of CARLA have been released since flamingo used it, including a jump to Unreal Engine 5, so there's likely to be API changes we need to conform to.   
  
  - [ ] Look at streamlining the inter-op between CARLA and SpSIM and update any post install scripts accordingly.

## Phase 4: Create datasets, challenges, and benchmarks.

A simulator will enable the creation of a standardized simulated dataset as well as accompanying benchmarks for a whole class of downstream tasks. In this phase we will have to select a handful of representative tasks where SPC are used or could be used and create benchmarks for these tasks. Possible tasks might include:

- High speed, low light SLAM

- On the fly stereo calibration

- Dynamic scene representations

- Structure from motion & 3D scanning

- Etc...

## Phase 5: Propose baseline methods. Write and publish paper.

Finally, we'll propose baseline approaches top solve these tasks given the proposed datasets and benchmarks, perform a few ablation studies explaining and characterizing the design decisions of the simulator (the impact of frame interpolation, lack of multi-bounce in transients, etc), and publish a manuscript detailing SpSIM. 

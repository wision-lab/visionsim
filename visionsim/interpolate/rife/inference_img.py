"""Modified from https://github.com/megvii-research/ECCV2022-RIFE/ to run on a sequence of images"""

from pathlib import Path

import imageio.v3 as iio
import torch
from natsort import natsorted
from torch.nn import functional as F

from visionsim.types import UpdateFn

from .RIFE_HDv3 import Model


def interpolate_img(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "**/*.png",
    input_files: list[Path] | None = None,
    model_dir: Path | None = None,
    exp: int = 4,
    update_fn: UpdateFn = None,
):
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    if input_files:
        img_paths = input_files
    else:
        img_paths = natsorted(input_dir.glob(pattern))

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    outpath = lambda p, i: Path(output_dir) / p.relative_to(input_dir).parent / f"{p.stem}_{i % 2**exp:02}{p.suffix}"

    if model_dir is None:
        # First check if it's in cwd or torch hub cache, else download
        if (Path(__file__).parent / "flownet.pkl").exists():
            model_dir = str(Path(__file__).parent)
        elif (Path(torch.hub.get_dir()) / "flownet.pkl").exists():
            model_dir = torch.hub.get_dir()
        else:
            print(f"Downloading weights to {str(Path(torch.hub.get_dir()) / 'flownet.pkl')}")
            Path(torch.hub.get_dir()).mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(
                "https://github.com/WISION-Lab/visionsim/releases/download/v0.1.0-alpha/flownet.pkl",
                str(Path(torch.hub.get_dir()) / "flownet.pkl"),
            )
            model_dir = torch.hub.get_dir()

    if len(img_paths) < 2:
        raise RuntimeError("At least two images required!")

    model = Model()
    model.load_model(model_dir)

    model.eval()
    model.device()

    # Set total number of steps
    if update_fn is not None:
        update_fn(total=len(img_paths) - 1)

    for _ in range(len(img_paths) - 1):
        # Skip ahead if all interpolated frames are already present
        p = Path(img_paths[0])
        if all(outpath(p, i).exists() for i in range(2**exp)):
            img_paths.pop(0)
            continue

        img0 = iio.imread(img_paths[0])
        img1 = iio.imread(img_paths[1])
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)

        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        img_list = [img0, img1]
        for i in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp

        for i in range(len(img_list)):
            # Don't save the last image in the img_list as it will become
            # the first one for the next interp interval, except if there isn't
            # a next interp interval
            if i == len(img_list) - 1:
                if not len(img_paths) <= 2:
                    continue
                else:
                    p = Path(img_paths[1])
            else:
                p = Path(img_paths[0])

            outpath(p, i).parent.mkdir(exist_ok=True, parents=True)
            iio.imwrite(outpath(p, i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        img_paths.pop(0)

        # Call any progress callbacks
        if update_fn is not None:
            update_fn(advance=1)

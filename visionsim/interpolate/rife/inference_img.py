"""Modified from https://github.com/megvii-research/ECCV2022-RIFE/ to run on a sequence of images"""

import argparse

import os
from pathlib import Path

from visionsim.types import UpdateFn

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


import cv2
import torch
from natsort import natsorted
from torch.nn import functional as F

from .RIFE_HDv3 import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate_img(img_paths, output_dir, model_dir=None, exp=4, ratio=0, rthreshold=0.02, rmaxcycles=8, update_fn: UpdateFn = None, **kwargs):
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    img_paths = natsorted(img_paths)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

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
    print(f"Found {len(img_paths)} images.")

    model = Model()
    model.load_model(model_dir)
    print("Loaded v3.x HD model.")

    model.eval()
    model.device()

    # Set total number of steps
    if update_fn is not None:
        update_fn(total=len(img_paths) - 1)

    for _ in range(len(img_paths) - 1):
        # Skip ahead if all interpolated frames are already present
        p = Path(img_paths[0])
        if all((Path(output_dir) / f"{p.stem}_{i%2**exp:02}{p.suffix}").exists() for i in range(2**exp)):
            img_paths.pop(0)
            continue

        if img_paths[0].endswith(".exr") and img_paths[1].endswith(".exr"):
            img0 = cv2.imread(img_paths[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img1 = cv2.imread(img_paths[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

        else:
            img0 = cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(img_paths[1], cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    middle_ratio = (img0_ratio + img1_ratio) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
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

            out = str(Path(output_dir) / f"{p.stem}_{i%2**exp:02}{p.suffix}")
            if img_paths[0].endswith(".exr") and img_paths[1].endswith(".exr"):
                cv2.imwrite(
                    out,
                    (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w],
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF],
                )
            else:
                cv2.imwrite(out, (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        img_paths.pop(0)

        # Call any progress callbacks
        if update_fn is not None:
            update_fn(advance=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation for a pair of images")
    parser.add_argument("--imgdir", dest="imgdir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--exp", default=4, type=int)
    parser.add_argument("--ratio", default=0, type=float, help="inference ratio between two images with 0 - 1 range")
    parser.add_argument(
        "--rthreshold", default=0.02, type=float, help="returns image when actual ratio falls in given range threshold"
    )
    parser.add_argument("--rmaxcycles", default=8, type=int, help="limit max number of bisectional cycles")
    parser.add_argument("--model", dest="model_dir", type=str, default=None, help="directory with trained model files")
    args = parser.parse_args()

    img_paths = [str(p) for p in natsorted(Path(args.imgdir).glob("*"))]
    interpolate_img(img_paths, **vars(args))

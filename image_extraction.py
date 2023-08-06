import sys

sys.path.insert(0, "./yolov7")
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_model(path: Path = None, gpu: bool = False):
    """
    Load model from .pt file
    Parameters
    ----------
    path: Where the .pt file is located
    gpu: To use gpu or not
    Returns
    -------
    A model variable
    """
    if path is None:
        path = Path(".") / "yolov7" / "yolov7.pt"
    if not Path.exists(path):
        raise FileNotFoundError("Can't find file yolov7.pt")
    else:
        if gpu:
            return torch.load(Path(path).as_posix())["model"]
        else:
            return torch.load(Path(path).as_posix(), map_location=torch.device("cpu"))[
                "model"
            ]


def hook_fn(m, i, o):
    """
    Hook function, use to debug when model is running
    Parameters
    ----------
    m: The model
    i: Input tensors
    o: Output tensors
    Returns
    -------
    Nothing
    """
    visualisation[m] = o


def get_image(path: str = None):
    """
    Get the image from path
    Parameters
    ----------
    path: Where the image is located
    Returns
    -------
    A img variable
    """
    if path is None:
        path = Path("sample_image.jpg").as_posix()
    if not Path.exists(Path(path)):
        raise FileNotFoundError("Can't find image file")
    im = cv2.imread(path)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1280, 1280))
    return img


def my_get_module(model, string):
    """
    Get exact module inside model by string
    Parameters
    ----------
    model: The model
    string: The string that use to locate the module (e.g. model.conv2d.0.weights)
    Returns
    -------
    A model variable
    """
    arr = string.split(".")[:-1]
    result = model
    for i in arr:
        if i.isdigit():
            result = result[int(i)]
        else:
            result = getattr(result, i)
    return result


def pre_process_model(model):
    """
    Pre process the model
    Parameters
    ----------
    model: The model
    Returns
    -------
    A model variable
    """
    for name, param in model.named_parameters():
        layer = my_get_module(model, name)
        layer.register_forward_hook(hook_fn)

    # Turn off recompute scale factor to avoid errors
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None

    return model


def process(img, model, gpu: bool = False):
    """
    Get output from image
    Parameters
    ----------
    img: Image as a cv2 variable
    model: The model
    gpu: Turn gpu on or off (MUST BE ON)
    Returns
    -------
    An output tensor
    """
    with autocast():
        with torch.no_grad():
            if gpu:
                outputs = model.forward(
                    torch.reshape(torch.tensor(img), (1, 3, 1280, 1280)).half().cuda()
                )
            else:
                outputs = model.forward(
                    torch.reshape(torch.tensor(img), (1, 3, 1280, 1280)).half()
                )
        return outputs


def convert(img, target_type_min, target_type_max, target_type):
    """
    Normalize the image to be outputed by cv2
    Parameters
    ----------
    img: Image
    target_type_min: Min value of converted matrix (should be 0)
    target_type_max: Max value of converted matrix (should be 255)
    target_type: Type of matrix (should be np.uint8)
    Returns
    -------
    A new image variable
    """
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def extract(output_dir: str = "Results", limit: int = 999):
    """
    Extract the images into output directory
    Parameters
    ----------
    output dir: Name of output folder
    limit: Limit of images per layer
    Returns
    -------
    Nothing
    """
    path = Path(".") / output_dir
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    # Save input image into output dir
    cv2.imwrite(Path(path / "input.png").as_posix(), img)
    # Get number of layers
    range_i = len(list(visualisation.keys()))
    for i in range(range_i):
        # Create folder
        layer_name = str(list(visualisation.keys())[i]).split("(")[0]
        path = Path(".") / output_dir / str(layer_name + "_" + str(i))
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        # Log the process
        print(f"layer: {i}/{range_i}")
        if torch.is_tensor(visualisation[list(visualisation.keys())[i]]):
            for j in range(
                min(list(visualisation[list(visualisation.keys())[i]].shape)[1], limit)
            ):
                im = convert(
                    np.float32(
                        visualisation[list(visualisation.keys())[i]][0, j, :, :]
                        .cpu()
                        .detach()
                        .numpy()
                    ),
                    0,
                    255,
                    np.uint8,
                )
                cv2.imwrite(Path(path / str(str(j) + ".png")).as_posix(), im)
    print("Extraction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image extraction file")
    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=False,
        default=Path(".") / "yolov7" / "yolov7.pt",
        help="Model path.",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        default="sample_image.jpg",
        help="image to be processed",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Folder that holds converted image(s)",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=999,
        help="limit of image exported in each layer",
    )
    parser.add_argument("-gpu", action="store_true", default=False, help="enable gpu")
    args = parser.parse_args()
    if not args.gpu:
        raise EnvironmentError(
            "GPU must be enabled, since some functions of yolo is not work with cpu"
        )
    visualisation = (
        {}
    )  # To store tensors, must be the same with variable inside hook_fn
    model = load_model(path=args.model_path, gpu=args.gpu)
    img = get_image(path=args.image_dir)
    model = pre_process_model(model)
    _ = process(img, model, gpu=args.gpu)
    if args.output_dir is None:
        folder_output_name = args.image_dir.split("/")[-1].split(".")[0]
    else:
        folder_output_name = args.output_dir
    extract(output_dir = folder_output_name, limit = args.limit)
    

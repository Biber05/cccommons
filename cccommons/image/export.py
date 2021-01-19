"""
    Exporter including writing image to filesystem or showing image on screen
"""
from typing import Union

import numpy as np
from cv2 import cv2


def write_img(img: np.ndarray, func: str, para: Union[int, str]):
    """
    export image to WIP directory including function and parameter as naming convention

    Args:
        img: image as numpy array
        func: function which was tested - (e.g. threshold)
        para: parameter for function (e.g. 127)

    Returns:
        write image to project folder with name "function_para.bmp" (e.g. threshold_127.bmp)
    """
    from cccommons.helpers.functions import name

    cv2.imwrite(name(func, para), img)


def show_img(img: np.ndarray, window_name="image"):
    """
    show image which opencv implementation

    Args:
        img: image as numpy array
        window_name: optional window name
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_hist(
    img: np.ndarray,
    channels=None,
    mask=None,
    ranges=None,
    bins: int = 256,
    normalize: bool = False,
):
    """
    Color BGR are correct
    Args:
        img:
        channels:
        mask:
        ranges:
        bins:
        normalize:

    Returns:

    """
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import hist

    for histogram in hist(img, channels, mask, ranges, bins, normalize):
        plt.plot(histogram)
        plt.xlim([0, bins])
    plt.show()


def hist2d(
    img: np.ndarray,
    channels: list,
    mask=None,
    ranges: tuple = (180, 256),
    bins: int = 256,
):
    r1, r2 = ranges
    from cccommons.image.process import to_hsv

    hsv = to_hsv(img)

    histogram = cv2.calcHist([hsv], channels, mask, [bins, bins], [0, r1, 0, r2])

    import matplotlib.pyplot as plt

    plt.imshow(histogram, interpolation="nearest")
    plt.show()


def plot_channels(img: np.ndarray, hsv: bool = False):
    from cccommons.image.process import to_hsv

    img = to_hsv(img) if hsv else img

    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    show_img(np.concatenate((h, s, v), axis=0), "channels")


def create_stream_from_folder(folder: str, ext: str = "tiff", fps=30):
    import os

    cap = cv2.VideoCapture(os.path.join(folder, "%1d.{}".format(ext)))
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter(
        os.path.join(folder, "converted.avi"), fourcc, fps, (640, 360)
    )

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break
    cap.release()
    out.release()

    return os.path.join(folder, "converted.avi")


__all__ = [
    "write_img",
    "show_img",
    "plot_hist",
    "hist2d",
    "plot_channels",
    "create_stream_from_folder",
]

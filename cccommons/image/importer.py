"""
    Importer including reading image from filesystem and taking an image using OpenCV.
    Also includes loader for sampling data.
"""
import os

import numpy as np
from cv2 import cv2


def read_img(filename: str, folder: str, gray=False) -> np.ndarray:
    """
    read image from given path
    specify if image should be imported in grayscale

    Args:
        filename: name of image file
        folder: folder (e.g. 1_trough)
        gray: boolean if the image should be converted to grayscale

    Returns:
        image in grayscale or color
    """
    return (
        cv2.imread(os.path.join(folder, filename), cv2.CV_8UC1)
        if gray
        else cv2.imread(os.path.join(folder, filename))
    )


__all__ = ["read_img"]

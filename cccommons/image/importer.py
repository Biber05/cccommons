"""
    Importer including reading image from filesystem and taking an image using OpenCV.
    Also includes loader for sampling data.
"""
import os
from time import sleep
from typing import Callable

import numpy as np
from cv2 import cv2

from cccommons.image.export import write_img
from cccommons.image.process import to_bgr, resize


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


def show_video():
    """
    show video from camera with given width and height set in config.py
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 800)  # 3
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)  # 4
    while True:
        ret, frame = cap.read()
        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break


def load_stream(filename, folder):
    """
    load video stream from given path

    Args:
        filename: name of video file
        folder: folder (e.g. 1_trough)

    Returns:
        image stream
    """
    cap = cv2.VideoCapture(os.path.join(folder, filename))
    while True:
        ret, frame = cap.read()
        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break


def capture_img_cv():
    """
    take and show image from camera

    Returns:
        image in color from camera
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return frame


def run_func_on_video(
    filename: str,
    folder: str,
    func,
    step_size: int = 1,
    timeout=0,
    capture_previous=False,
    write: bool = False,
):
    """
    calls a function on every n frame in video

    if capture_previous:
        func(frame, previous)
    else:
        func(frame)

    Args:
        filename: video filename
        folder: video dir
        func: function to be called
        step_size: calls function every n frames - to improve performance
        timeout: timeout between two images - debugging
        capture_previous: boolean if capture should save previous image and provide it to function as well
        write: boolean if should write every n (step_size) image to file system

    Returns:
        VideoStream
    """
    cap = cv2.VideoCapture(os.path.join(folder, filename))
    index = 1
    ret, frame = cap.read()
    previous = frame
    while True:
        ret, frame = cap.read()
        if ret:
            if index % step_size == 0:
                if capture_previous:
                    img = func(frame, previous)
                else:
                    img = func(frame)
                if write:
                    write_img(img, func.__name__, index)
                img = to_bgr(img)
            else:
                img = frame
            previous = frame

            frame = to_bgr(frame)
            frame = cv2.resize(src=frame, dsize=(img.shape[1], img.shape[0]))
            frame = np.concatenate((frame, img), axis=1)

            cv2.imshow("frame", frame)
            index += 1
            sleep(timeout)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def concat_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img1 = to_bgr(img1)
    img2 = to_bgr(img2)
    img2 = cv2.resize(src=img2, dsize=(img1.shape[1], img1.shape[0]))
    frame = np.concatenate((img1, img2), axis=1)
    return frame


def run_func_on_camera(func, path: str = None):
    """
    calls a function on every frame on camera image

    if path is None:
        camera is used
        video will be resized
    else:
        read video from path

    Args:
        func: function to be called
        path: filename to video

    Returns:
        VideoStream
    """
    if path is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        previous = resize(frame, 0.2)
    else:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        previous = frame
    while True:
        ret, frame = cap.read()
        if path is not None:
            frame = resize(frame, 0.2)
        if ret:
            cv2.imshow("frame", func(frame, previous))
            previous = frame
            sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_func_in_windows(img: np.ndarray, func, columns: int = 4, rows: int = 4):
    """
    Run a function in a sliding window.

    Args:
        img: image as numpy array
        func: function
        columns: number of columns
        rows: number of rows

    Returns:
        image as numpy array AND return value of function
    """
    h, w, _ = img.shape

    w_step = int(w / columns)
    h_step = int(h / rows)

    for c in range(0, columns):
        for r in range(0, rows):
            func(img[r * h_step : (r + 1) * h_step, c * w_step : (c + 1) * w_step])
    return img


def run_plot_in_video(func: Callable, path: str, capture_previous: bool = False):
    data = []

    cap = cv2.VideoCapture(path)
    index = 1

    ret, frame = cap.read()
    previous = frame

    while True:
        ret, frame = cap.read()
        if ret:
            if capture_previous:
                value = func(frame, previous)
            else:
                value = func(frame)
            data.append([index, value])
            cv2.imshow("frame", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        index += 1
    cap.release()
    cv2.destroyAllWindows()

    import pandas as pd
    from cccommons.plot.uni import Plot

    pl = Plot(
        pd.DataFrame(data), suffix=path.split("/")[-1].split(".")[0], plot_dir=path
    )
    pl.hist2d_line(x=0, y=1)


__all__ = ["read_img"]

"""
    Image processing which is a wrapper of standard OpenCV functions and algorithms.
    Common parameters and pre-processing steps are already encapsulated.
"""
from typing import Union

import numpy as np
from cv2 import cv2


def resize(img: np.ndarray, factor: float, interpolation=cv2.INTER_LANCZOS4):
    """
    resize image by given factor

    Args:
        img: image as numpy array
        factor: 0.5 reduces height and width by half
        interpolation: Default ist INTER_LANCZOS4

    Returns:
         img
    """
    return cv2.resize(
        src=img, dsize=None, fx=factor, fy=factor, interpolation=interpolation
    )


def crop(img: np.ndarray, contour) -> np.ndarray:
    """
    crop image by a given contour.
    uses a bounding rectangle without rotation.

    Args:
         img: image as numpy array
         contour: contour which determine ROI for cropping

    Returns:
         img
    """
    x, y, w, h = cv2.boundingRect(contour)
    return img[y : y + h, x : x + w]


def crop_with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    crop image by a given mask.

    Args:
        mask: mask as numpy array
        img: image as numpy array

    Returns: img
    """
    contour = find_contours(mask)[0]
    return crop(img, contour)


def crop_with_rect(img: np.ndarray, rect) -> np.ndarray:
    """
    crop image by a given mask.

    Args:
        img: image as numpy array
        rect: tuple including x, y position, width and height

    Return:
        img
    """
    x, y, w, h = rect
    x, y, w, h = int(x), int(y), int(w), int(h)
    return img[y : y + h, x : x + w]


def is_colored(img: np.ndarray) -> bool:
    """
    checks is a image is in grayscale or colored

    Args:
        img: image as numpy array

    Returns:
        True if image is in color and False if it is grayscale
    """

    return True if len(img.shape) == 3 else False


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    converts image from BGR to GRAY
    if image is already in grayscale then do nothing

    Args:
        img: image as numpy array

    Returns:
        img in grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_colored(img) else img


def to_hsv(img: np.ndarray) -> np.ndarray:
    """
    to hsv color space

    Args:
        img: image as numpy array

    Returns:
        img in HSV color space
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_bgr(img: np.ndarray) -> np.ndarray:
    if is_colored(img):
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def avg_pixel(img: np.ndarray):
    """
    average pixel intensity across the entire image

    Args:
        img: image as numpy array

    Returns:
        tuple including the avg for each color space / dimension
    """
    return cv2.mean(img)


def mean_pixel(img: np.ndarray) -> Union[tuple, np.ndarray]:
    if is_colored(img):
        b = np.median(img[0])
        g = np.median(img[1])
        r = np.median(img[2])
        return b, g, r
    else:
        return np.median(img)


def thresh(
    img: np.ndarray,
    threshold=127,
    max_value=255,
    kernel_size=3,
    thresh_type=cv2.THRESH_BINARY_INV,
) -> np.ndarray:
    """
    creates a binary image for given threshold value in range 0-255

    Args:
        img: image as numpy array
        threshold: value which splits colors into binary values
        max_value: after splitting upper value is set to this grayscale
        kernel_size: kernel size of matrix
        thresh_type: THRESH_BINARY_INV, THRESH_BINARY, ...

    Returns:
        binary image
    """
    img = to_gray(img)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return cv2.threshold(img, threshold, max_value, thresh_type)[1]


def gaussian(img: np.ndarray, kernel_size=3) -> np.ndarray:
    """
    gaussian blur filter using matrix of size "kernel"

    Args:
        img: image as numpy array
        kernel_size: size of NxN matrix

    Returns:
        img
    """
    img = to_gray(img)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def laplace(img: np.ndarray) -> np.ndarray:
    """
    laplace filter

    Args:
        img: image as numpy array

    Returns:
        img
    """
    return cv2.Laplacian(img, cv2.CV_64F)


def sobel_x(img: np.ndarray, kernel_size=3) -> np.ndarray:
    """
    sobel operator in x direction to identify vertical gradients

    Args:
        img: image as numpy array
        kernel_size: size of NxN matrix

    Returns:
        img
    """
    return cv2.Sobel(to_gray(img), cv2.CV_64F, 1, 0, ksize=kernel_size)


def sobel_y(img: np.ndarray, kernel_size=3) -> np.ndarray:
    """
    sobel operator in y direction to identify horizontal gradients

    Args:
        img: image as numpy array
        kernel_size: size of NxN matrix

    Returns:
        img
    """
    return cv2.Sobel(to_gray(img), cv2.CV_64F, 0, 1, ksize=kernel_size)


def canny(img: np.ndarray, thresh_1=50, thresh_2=150, aperture_size=3) -> np.ndarray:
    """
    canny edge detector

    Args:
        img: image as numpy array
        thresh_1: minimum threshold for canny detector
        thresh_2: backward threshold for canny detector
        aperture_size: kernel size

    Returns:
        list of edges
    """
    gray = to_gray(img)
    img = np.uint8(gray)
    return cv2.Canny(img, thresh_1, thresh_2, apertureSize=aperture_size)


def split(img: np.ndarray) -> [np.ndarray]:
    """
    Splits image into his channels

    Args:
        img: image as numpy array

    Returns:
        [R, G, B]
    """
    if not is_colored(img):
        raise BaseException("Should be a colored image. Got grayscale instead.")

    b, g, r = cv2.split(img)
    return [r, g, b]


def color_range(img: np.ndarray, lower, upper) -> np.ndarray:
    """
    generally a binary image with two thresholds of BGR

    Args:
        img: image as numpy array
        lower: tuple of BGR colors as lower bound
        upper: tuple of BGR colors as upper bound

    Returns:
        img: image as numpy array
    """
    return cv2.inRange(img, lower, upper)


def equalize(img: np.ndarray) -> np.ndarray:
    """

    Args:
        img: image as numpy array

    Returns:
        image
    """
    if is_colored(img):
        equ_b = cv2.equalizeHist(get_color(img, "b"))
        equ_g = cv2.equalizeHist(get_color(img, "g"))
        equ_r = cv2.equalizeHist(get_color(img, "r"))
        return cv2.merge((equ_b, equ_g, equ_r))
    else:
        return cv2.equalizeHist(img)


def get_color(img: np.ndarray, color: str) -> np.ndarray:
    """

    Args:
        img: image as numpy array
        color: r g or b as string value

    Returns:
        image with one channel
    """
    if color == "r":
        return img[:, :, 0]
    elif color == "g":
        return img[:, :, 1]
    elif color == "b":
        return img[:, :, 2]
    else:
        raise ValueError("Wrong color - use 'r' 'g' or 'b'")


def gamma(img: np.ndarray, gamma_value=1.0):
    """
    use a lookup table to calculate new gamma value of pixels

    Args:
        img: image as numpy array
        gamma_value: factor to darken or lighten image
            > 1.0 lighten, <1.0 darken, =1.0 no changes

    Returns:
        image with adjusted gamma values
    """
    inverted = 1.0 / gamma_value
    table = np.array(
        [((i / 255.0) ** inverted) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def invert_image(img: np.ndarray) -> np.ndarray:
    """
    inverts an binary image
    black -> white
    white -> black

    Args:
        img: image as numpy array

    Returns:
        inverted image as numpy array
    """
    img = to_gray(img)
    return cv2.bitwise_not(img)


def kernel(size):
    """
    creates a kernel with ones of size NxN

    Args:
        size: kernel size

    Returns:
        :return: kernel of size NxN
    """
    return np.ones((size, size))


def dilate(img: np.ndarray, rounds=1, size=3) -> np.ndarray:
    """
    strengthen white values

    Args:
        img: image as numpy array
        rounds: iterations of operation
        size: size of kernel

    Returns:
        img: dilated image as numpy array
    """
    img = to_gray(img)
    return cv2.dilate(
        img,
        kernel(size),
        iterations=rounds,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=cv2.BORDER_DEFAULT,
    )


def erode(img: np.ndarray, rounds=1, size=3) -> np.ndarray:
    """
    strengthen black values

    Args:
        img: image as numpy array
        rounds: iterations of operation
        size: size of NxN matrix

    Returns:
        img
    """
    img = to_gray(img)
    return cv2.erode(img, kernel(size), iterations=rounds)


def closing(img: np.ndarray, size=3, iterations=1) -> np.ndarray:
    """
    dilate -> erode

    Args:
        iterations: number of iterations
        img: image as numpy array
        size: size of NxN matrix

    Returns:
        img
    """
    img = to_gray(img)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel(size), iterations=iterations)
    return img


def opening(img: np.ndarray, size=3, iterations=1) -> np.ndarray:
    """
    erode -> dilate

    Args:
        iterations: number of iterations
        img: image as numpy array
        size: size of NxN matrix

    Returns:
        img
    """
    img = to_gray(img)
    img = cv2.morphologyEx(
        src=img, op=cv2.MORPH_OPEN, kernel=kernel(size), iterations=iterations
    )
    return img


def apply_mask(img: np.ndarray, mask) -> np.ndarray:
    """
    arithmetic operation for images
    e.g. AND operation with a binary image like a mask

    Args:
        img: image as numpy array
        mask: mask as numpy array

    Returns:
        combination of mask and img
    """
    return cv2.copyTo(src=img, mask=mask)


def create_mask_from_rect(rect: tuple) -> np.ndarray:
    mask = np.zeros(shape=(360, 640, 3), dtype=np.uint8)
    x, y, w, h = rect
    p1, p2 = (x, y), (x + w, y + h)
    mask = cv2.rectangle(mask, p1, p2, color=(255, 255, 255), thickness=-1)
    return invert_image(mask)


def find_contours(img: np.ndarray):
    """
    find contours of given image

    Args:
        img: image as numpy array

    Returns:
        list of contours without the entire image
    """
    img = thresh(img)

    # independent from OpenCV Version - sort of tuple is different between major releases
    temp = cv2.findContours(
        image=img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE
    )
    contours = temp[0] if len(temp) == 2 else temp[1]

    return contours[1:]


def draw_contours(img: np.ndarray, contours) -> np.ndarray:
    """
    draw bounding rectangle of provided contour

    Args:
        img: image as numpy array
        contours: list of contours

    Returns:
        image including contours as rectangle
    """
    # If zero contours were found - no image will be generated
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # draw rectangle for each contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 127, 0), 3)
    return img


def find_and_draw_contours(img: np.ndarray) -> np.ndarray:
    contours = find_contours(img)
    return draw_contours(img, contours)


def draw_rect(
    img: np.ndarray, dimensions: tuple, color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    draw rectangle of position and size

    Args:
        img: image as numpy array
        dimensions: x and y coordinate as tuple (x,y) and width and height of rectangle (width, height) - (x,y,w,h)
        color: color of rectangle as BGR tuple (b,g,r) - default (255,255,255) = white

    Returns:
        image including contours as rectangle
    """
    x, y, w, h = dimensions

    cv2.rectangle(
        img,
        (int(x), int(y)),
        (int(x) + int(w), int(y) + int(h)),
        color=color,
        thickness=1,
    )
    return img


def filter_contours(contours, min_size=0, max_size=500) -> []:
    """
    reduces contours by given min length and max length
    e.g. size of barcode

    Args:
        contours: list of contours
        min_size: minimum area of contour
        max_size: maximum area of contour

    Returns:
        reduced list of contours
    """
    return list(
        filter(lambda cnt: min_size < cv2.contourArea(cnt) < max_size, contours)
    )


def write_text(
    img: np.ndarray, text: Union[list, str, dict], position: tuple = (10, 30)
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_COMPLEX
    color = (0, 255, 0)
    size = 0.5
    line_type = 2
    if isinstance(text, str):
        cv2.putText(img, text, position, font, size, color, line_type)
    elif isinstance(text, list):
        for i, txt in enumerate(text):
            cv2.putText(
                img,
                str(txt),
                (position[0], position[1] + i * 20),
                font,
                size,
                color,
                line_type,
            )
    else:
        for i, label in enumerate(text):
            cv2.putText(
                img,
                f"{label} : {text[label]}",
                (position[0], position[1] + i * 20),
                font,
                size,
                color,
                line_type,
            )
    return img


__all__ = [
    "resize",
    "crop",
    "crop_with_mask",
    "crop_with_rect",
    "is_colored",
    "to_gray",
    "to_bgr",
    "to_hsv",
    "avg_pixel",
    "mean_pixel",
    "thresh",
    "gaussian",
    "laplace",
    "sobel_x",
    "sobel_y",
    "canny",
    "split",
    "color_range",
    "equalize",
    "get_color",
    "gamma",
    "invert_image",
    "kernel",
    "dilate",
    "erode",
    "closing",
    "opening",
    "apply_mask",
    "create_mask_from_rect",
    "find_contours",
    "filter_contours",
    "draw_contours",
    "find_and_draw_contours",
    "draw_rect",
    "write_text",
]

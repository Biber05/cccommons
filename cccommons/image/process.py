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
    return img[y: y + h, x: x + w]


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
    return img[y: y + h, x: x + w]


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


def diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    difference between two images to detect changes over time

    Args:
        img1: image as numpy array
        img2: image as numpy array

    Returns:
        combined image
    """
    return cv2.absdiff(img1, img2)


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


def hough(
        img: np.ndarray,
        min_line_length=100,
        max_line_gap=10,
        source: np.ndarray = None,
        draw=True,
) -> np.ndarray:
    edges = canny(img)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is not None:
        if draw:
            if source is not None:
                source = draw_lines(source, lines)
            else:
                img = draw_lines(img, lines)
    return img if source is None else source


def hough_lines(img: np.ndarray, min_line_length=100, max_line_gap=10) -> list:
    """
    Detect lines using hough transformation

    Args:
        img: image as numpy array
        min_line_length: minimal length of detected line
        max_line_gap: max gap between lines segments to treat them as single line

    Returns:
        list of lines
    """
    edges = canny(img)
    result = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return result if result is not None else []


def angle(p1, height: int) -> int:
    import math

    p2 = (0, height)
    p1 = (p1[1] - p1[3], p1[0] - p1[2])

    def dot_product(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dot_product(v, v))

    return int(np.rad2deg(math.acos(dot_product(p1, p2) / (length(p1) * length(p2)))))


def draw_lines(img: np.ndarray, lines: list):
    """
    Draw lines on image
    Args:
        img: image as numpy array
        lines: list of line coords

    Returns:
        image
    """
    height = img.shape[0]
    if lines is not None:
        for line in lines:
            a = angle(line, height)
            if 60 < a < 160:
                x1, y1, x2, y2 = line
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # norm_x, norm_y = x2 - x1, y2 - y1
                # cv2.line(img, (0, 0), (norm_x, norm_y), (0, 0, 255), 2)
    return img


def match_template(img: np.ndarray, template, threshold=0.5) -> np.ndarray:
    """
    finds a template in a picture using matchTemplate.
    does not work for rotated or scaled versions of the template

    Args:
        img and template in grayscale

    Returns:
        img including found match
    """
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    return img


def raster(img: np.ndarray, cols: int, rows: int) -> [np.ndarray]:
    """
    Create a raster of an image and returns a list of ROI

    Args:
        img: image as numpy array
        cols: number of columns
        rows: number of rows

    Returns:

    """
    img_list = []

    h, w, _ = img.shape

    w_step = int(w / cols)
    h_step = int(h / rows)

    for c in range(0, cols):
        for r in range(0, rows):
            img_list.append(
                img[r * h_step: (r + 1) * h_step, c * w_step: (c + 1) * w_step]
            )
    return img_list


def sliding_window(
        img: np.ndarray, size: tuple = (32, 32), step_size: tuple = (16, 16)
):
    """
    Sliding window of a image for given kernel and step size

    Args:
        img: image as numpy array
        size: size of kernel as tuple - should be (2^x, 2^y)
        step_size: moving of windows as tuple for x and y direction - should be (2^n, 2^m)

    Returns:
        Iterator returning x, y and image
    """
    for y in range(0, img.shape[0], step_size[1]):
        for x in range(0, img.shape[1], step_size[0]):
            # yield the current window
            yield x, y, img[y: y + size[1], x: x + size[0]]


def draw_points(
        img: np.ndarray, points: list, color: tuple = (255, 0, 0)
) -> np.ndarray:
    """
    Draws points from a list of coordinates

    Args:
        img: image as numpy array
        points: list of tuples (x,y)
        color: BGR color triple

    Returns:
        return image including drawn points
    """
    points = np.array(points, dtype=np.uint8)

    for i in points:
        x, y = i.ravel()
        cv2.circle(img, (x, y), radius=3, color=color, thickness=-1)

    return img


def optical_flow(img: np.ndarray, previous: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape=img.shape, dtype=np.uint8)
    mask[..., 1] = 255

    gray = to_gray(img)
    previous = to_gray(previous)

    flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, polar_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = polar_angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb


def gradient_diff(img: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """
    Gradient using difference image
    Args:
        img: image as numpy array at time t0
        previous: image as numpy array at time t-1

    Returns:
        gradient diff image
    """
    diff_img = diff(img, previous)
    return gradient(diff_img)


def gradient(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    calculates the magnitude and polar_angle of 2D vectors using Sobel operator on both axis.
    Args:
        img: image as numpy array
        kernel_size: kernel size of Sobel operator (NxN)

    Returns:
        magnitude of gradients in image shape
    """
    img = np.float32(img) / 255.0

    sy = sobel_y(img, kernel_size=kernel_size)
    sx = sobel_x(img, kernel_size=kernel_size)

    mag, polar_angle = cv2.cartToPolar(sx, sy, angleInDegrees=True)

    mag = np.ndarray.astype(mag, dtype=np.float32)
    mag = to_bgr(mag)

    return mag


def get_bounding_rect_from_contour(contour) -> tuple:
    """
    creates a tuple including position and dimension of a contour

    Args:
        contour

    Returns:
        :return tuple of position and dimension
    """
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y), (w, h)


def calc_moments(contours, min_value=0, max_value=640) -> []:
    """
    moment: weighted average of pixel intensity

    Args:
        contours: list of contours
        min_value: minimum area of contour
        max_value: maximum area of contour

    Returns:
        :return list of moments from contours
    """
    M = []
    for c in contours:
        if min_value <= cv2.contourArea(c) <= max_value:
            M.append(cv2.moments(c))
    return M


def centroids(moments: []) -> []:
    """
     calc centroids of moments

    Args:
        moments list of moments

    Returns:
        list of centroids
    """
    C = []
    for m in moments:
        C.append(calc_centroid(m))
    return C


def calc_centroid(moment) -> tuple:
    """
    physical center of given area

    Args:
        moment

    Returns:
        tuple of x and y coordinates (x,y)
    """
    cx = int(moment["m10"] / moment["m00"])
    cy = int(moment["m01"] / moment["m00"])
    return cx, cy


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
    "sliding_window",
    "gradient",
    "angle",
    "diff",
    "calc_centroid",
    "centroids",
    "optical_flow",
]

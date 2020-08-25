import cv2
import random
import numpy as np

from pathlib import Path
from scipy import interpolate
from typing import List, Tuple, Iterable, Set


def generate_art(in_path: Path, out_path: Path, sample=0.5, max_dim=500) -> None:
    print(f"Processing image '{in_path}'...")
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)

    # resize to max dimension
    h, w = img.shape[:2]
    if h > w:
        r = max_dim / float(h)
        dim = (int(w * r), max_dim)
    else:
        r = max_dim / float(w)
        dim = (max_dim, int(h * r))
    resized = cv2.resize(img, dim)
    (w, h) = dim

    # smooth image
    filtered = cv2.bilateralFilter(resized, 7, 50, 50)

    # calculate canny thresholds
    print("Edge extraction...")
    high_thresh, thresh_im = cv2.threshold(
        filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lowThresh = 0.5 * high_thresh

    otsu = cv2.Canny(filtered, lowThresh, high_thresh)
    cv2.imwrite(str(out_path / f"edges_{in_path.name}"), otsu)
    print("Done!")

    # obtain edge points
    indices = np.where(otsu != [0])
    edge_points: Set[Tuple[int, int]] = set()

    points = list(zip(indices[0], indices[1]))
    print(f"Points before reduction: {len(points)}")
    points = random.sample(points, int(len(points) * sample))
    print(f"Points after reduction: {len(points)}")

    print("Calculating point sequence...")
    point_order = get_sequence_naive(points)
    print("Done!")

    print("Fitting spline to points...")
    # transform coordinate system
    x = [point[1] for point in point_order]
    y = [h - point[0] for point in point_order]

    tck, u = interpolate.splprep([x, y], s=0)
    u = np.linspace(0, 1, num=len(tck[0]) * 100, endpoint=True)
    out = interpolate.splev(u, tck)
    print("Done!")

    # TODO generate corresponding Bézier curves with Boehm's algorithm
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.insert.html

    # TODO use Bézier curves to create nice SVG

    print("Creating output image...")
    fg_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    rx = out[0]
    ry = out[1]

    out_img = np.full((h, w, 3), bg_color, np.uint8)

    print("Drawing spline on output...")
    xy = list(zip(rx, ry))
    progress_steps = list(range(0, 101, 10))
    progress = [len(xy) * step * 0.01 for step in progress_steps]

    for index, point in enumerate(xy):
        if index >= progress[0]:
            print(f"  Progess: {progress_steps[0]}% - {index}/{len(xy)}")
            progress.pop(0)
            progress_steps.pop(0)
        # print(f"Setting ({h-int(point[1])}, {int(point[0])}) to red")
        x = h - round(point[1])
        y = round(point[0])
        if 0 <= x < h and 0 <= y < w:
            out_img[x, y] = fg_color
    print("Done!")

    kernel = np.ones((2, 2), np.uint8)
    # out_img = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, kernel)
    out_img = cv2.dilate(out_img, kernel, iterations=1)
    out_img = cv2.GaussianBlur(out_img, (5, 5), 0)

    cv2.imwrite(str(out_path / in_path.name), out_img)

    print("-" * 20)
    # cv2.imshow("input", np.hstack([base, edges]))
    # cv2.imshow("ART", np.hstack([image, blank_image]))
    # cv2.waitKey(0)


def get_sequence_naive(points: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    def get_closest(
        start: Tuple[int, int], points: Iterable[Tuple[int, int]]
    ) -> Tuple[int, int]:
        min_dist = float("inf")
        closest = None

        for target in points:
            dist = (target[0] - start[0]) ** 2 + (target[1] - start[1]) ** 2
            if dist < min_dist:
                closest = target
                min_dist = dist
        return closest

    curr_point: Tuple[int, int] = points.pop()
    sequence: List[Tuple[int, int]] = []

    progress_steps = list(range(0, 101, 10))
    progress = [len(points) * step * 0.01 for step in progress_steps]

    while points:
        if len(sequence) >= progress[0]:
            print(f"  Progess: {progress_steps[0]}% - {len(sequence)}/{len(points)}")
            progress.pop(0)
            progress_steps.pop(0)
        closest = get_closest(curr_point, points)
        points.remove(closest)
        sequence.append(closest)
        curr_point = closest
    return sequence


if __name__ == "__main__":
    input_dir = Path("examples")
    output_dir = Path("examples/out")
    img_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for path in input_dir.iterdir():
        if path.suffix in img_formats:
            generate_art(path, output_dir)
            # generate_line_art(filename, input_dir, output_dir)


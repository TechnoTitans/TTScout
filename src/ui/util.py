import enum

from scipy.spatial import distance as dist
import numpy as np
import cv2


# TODO is there a better way of doing this or is this already in an lib somewhere
# TODO also don't like how I have to use the rgb property, I wish it would just be
#  Color.RED and it would just pass in a tuple but you'd have to do either
#  Color.RED.value or in this case (somewhat better) Color.RED.rgb


class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    BLACK = (0, 0, 0)
    ALMOST_BLACK = (30, 30, 30)

    @property
    def rgb(self):
        return self.value

    @property
    def hex(self):
        return '#%02x%02x%02x' % self.value


def order_points(pts):
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    leftmost = x_sorted[:2, :]
    rightmost = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates, so we can grab the top-left and bottom-left
    # points, respectively
    leftmost = leftmost[np.argsort(leftmost[:, 1]), :]
    (tl, bl) = leftmost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    d = dist.cdist(tl[np.newaxis], rightmost, "euclidean")[0]
    (br, tr) = rightmost[np.argsort(d)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def transform_dimension(x0, x1, x2, x3):
    return np.sqrt(
        ((x0[0] - x1[0]) ** 2) + ((x0[1] - x1[1]) ** 2)
    ), np.sqrt(
        ((x2[0] - x3[0]) ** 2) + ((x2[1] - x3[1]) ** 2)
    )


def transform_width_height_compute(tl, tr, br, bl):
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    width_a, width_b = transform_dimension(br, bl, tr, tl)
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a, height_b = transform_dimension(tr, br, tl, bl)
    max_height = max(int(height_a), int(height_b))

    return max_width, max_height


def four_point_transform(image: np.ndarray, pts) -> np.ndarray:
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width and height of the new image
    max_width, max_height = transform_width_height_compute(tl, tr, br, bl)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    # return the warped image
    return warped

from __future__ import annotations

import time
from typing import Any, Callable, TextIO

import math
import json
import tkinter as tk
import warnings
import atexit
import threading
import enum
import os.path
import csv
import random

import cv2
import numpy as np
import pupil_apriltags
from PIL import Image, ImageTk
from pupil_apriltags import Detector
from scipy import ndimage as ndi

from util import four_point_transform, Color, rotate_image, filter_out_shadows
from scroll_frame import ScrollFrame
from canvas import Canvas


def get_capture_device(source: int, suppress_warn=False):
    device = cv2.VideoCapture(source)
    if device is None or not device.isOpened():
        if not suppress_warn:
            warnings.warn(f"Unable to open VideoCapture Stream on Source: {source}")
        return None

    # set to maximum resolution
    # device.set(3, device.get(cv2.CAP_PROP_FRAME_WIDTH))
    # device.set(4, device.get(cv2.CAP_PROP_FRAME_HEIGHT))

    device.set(3, 1280)
    device.set(4, 720)

    return device


def clear_widget(parent: tk.BaseWidget, exclude: tk.BaseWidget | [tk.BaseWidget] = None):
    if parent.winfo_exists():
        for widget in parent.winfo_children():
            if (type(exclude) is list and widget not in exclude) or (widget is not exclude):
                widget.destroy()


def clear_window(window: tk.Tk):
    if window.winfo_exists():
        for widget in window.winfo_children():
            widget.destroy()


def get_tag_bounds(tag_pts):
    # Find the top-left and bottom-right corners of the box
    x_vals, y_vals = zip(*tag_pts)
    x_min, y_min = np.min(x_vals), np.min(y_vals)
    x_max, y_max = np.max(x_vals), np.max(y_vals)

    return x_min, y_min, x_max, y_max


def get_center(x_min, y_min, x_max, y_max):
    # Return the centers of the tag
    return int((x_min + x_max) / 2), int((y_min + y_max) / 2)


def sum_tup(tup_a, tup_b):
    return tuple(map(sum, zip(tup_a, tup_b)))


def consecutive(data, step_size=1):
    return np.split(data, np.where(np.diff(data) != step_size)[0] + 1)


def pt_inside_rect(rx0: int, ry0: int, rx1: int, ry1: int, px: int, py: int) -> bool:
    return rx0 <= px <= rx1 and ry0 <= py <= ry1


class CameraApp:
    PROCESSING_DIM = (640, 480)
    ORIENTATION_TAG_ID = 0

    PROCESSING_CLIP_LIMIT = 2.0
    PROCESSING_TILE_GRID_SIZE = (8, 8)

    BUBBLE_DETECTION_BASELINE = 0.2

    _TK_TEXT_NORMAL = "normal"
    _TK_TEXT_DISABLED = "disabled"

    class Position(enum.Enum):
        TL = enum.auto()
        TR = enum.auto()
        BL = enum.auto()
        BR = enum.auto()

    def __init__(self, window: tk.Tk, device: cv2.VideoCapture = None):
        self.window = window
        self.window.title("TTScout")

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.snapshot_frame = None

        self.snapshot_button = tk.Button(self.window, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.cap = device if device is not None else get_capture_device(0)
        if self.cap is None:
            exit(1)

        self.video_stream = tk.Label(self.video_frame)
        self.video_stream.pack()

        self.preview_window = tk.Toplevel(self.window)
        self.preview_window.withdraw()

        self.tag_id_text_offset = (10, -10)
        self.tag_layout = [
            [1, 6],
            [8, 3],
        ]

        n_tag_rows = len(self.tag_layout)
        n_top_row = len(self.tag_layout[0])
        n_bottom_row = len(self.tag_layout[n_tag_rows - 1])

        self.pos_indexes = {
            self.Position.TL: (0, 0),
            self.Position.TR: (0, n_top_row - 1),
            self.Position.BL: (n_tag_rows - 1, 0),
            self.Position.BR: (n_tag_rows - 1, n_bottom_row - 1)
        }

        self.possible_tags = [tag_id for tags in self.tag_layout for tag_id in tags]

        self.detector = Detector(
            families='tag16h5',
            nthreads=1,
            quad_decimate=1,
            quad_sigma=1,
            refine_edges=10,
            decode_sharpening=0.25,
            debug=0
        )

    def img_window(self, images: [Image], dims=(240, 320)):
        self.preview_window.protocol("WM_DELETE_WINDOW", self.retry_preview)

        # Create a list to store PhotoImage objects
        photos = []

        # Iterate through the images and create a PhotoImage object for each one
        for img in images:
            img = img.resize(dims, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            photos.append(photo)

        # Display the images in a vertical stack
        if self.preview_window.winfo_exists():
            self.preview_window.deiconify()

        preview_frame = tk.Frame(self.preview_window)
        preview_frame.pack(side=tk.TOP, padx=10, pady=10)

        for photo in photos:
            preview_label = tk.Label(preview_frame, image=photo)
            preview_label.image = photo  # Keep reference to avoid garbage collection
            preview_label.pack(side=tk.TOP)

        confirm_button = tk.Button(self.preview_window, text="Confirm", command=self.confirm_preview)
        confirm_button.pack(side=tk.LEFT)

        retry_button = tk.Button(self.preview_window, text="Retry", command=self.retry_preview)
        retry_button.pack(side=tk.RIGHT)

    def get_tag_at_pos(self, tags: [pupil_apriltags.Detection], position: Position) -> pupil_apriltags.Detection | None:
        i, j = self.pos_indexes[position]
        tag_id = self.tag_layout[i][j]

        for tag in tags:
            if tag.tag_id == tag_id:
                return tag

        return None

    def get_tags_at_pos(self,
                        tags: [pupil_apriltags.Detection],
                        positions: [Position]) -> [pupil_apriltags.Detection | None]:
        return [self.get_tag_at_pos(tags, position) for position in positions]

    def orientation_correction(self, image: np.ndarray) -> np.ndarray:
        tags = self.detector.detect(image)

        tag_pts = np.array([tag.center for tag in tags if tag.tag_id in self.possible_tags])
        orientation_tag: pupil_apriltags.Detection = next(
            (tag for tag in tags if tag.tag_id == CameraApp.ORIENTATION_TAG_ID), None
        )

        if orientation_tag is None:
            return image

        centroid_pt = tag_pts.mean(axis=0)
        centroid_x, centroid_y = centroid_pt

        mat_shape = image.shape
        mat_center_x, mat_center_y = mat_shape[1] / 2, mat_shape[0] / 2

        center_x = centroid_x if math.isclose(centroid_x, mat_center_x, abs_tol=5) else mat_center_x
        center_y = centroid_y if math.isclose(centroid_y, mat_center_y, abs_tol=5) else mat_center_y

        tag_x, tag_y = orientation_tag.center

        is_x_center = math.isclose(tag_x, center_x, abs_tol=10)
        is_y_center = math.isclose(tag_y, center_y, abs_tol=10)

        if tag_x < center_x and is_y_center:
            # on left side
            desired_rot_angle = -90
        elif tag_x > center_x and is_y_center:
            # on right side
            desired_rot_angle = 90
        elif tag_y < center_y and is_x_center:
            # on top
            desired_rot_angle = 0
        elif tag_y > center_y and is_x_center:
            # on bottom
            desired_rot_angle = 180
        else:
            desired_rot_angle = 0

        return rotate_image(image, desired_rot_angle)

    def process_snapshot(self, grayscale: np.ndarray) -> np.ndarray:
        clahe: cv2.CLAHE = cv2.createCLAHE(
            clipLimit=self.PROCESSING_CLIP_LIMIT, tileGridSize=self.PROCESSING_TILE_GRID_SIZE
        )

        equalized_img = clahe.apply(grayscale)

        shadow_filtered_image = filter_out_shadows(equalized_img)
        ret, threshold = cv2.threshold(
            shadow_filtered_image, np.median(shadow_filtered_image), 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        return threshold

    def contour_apriltags(self, image: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        tag_pts = [tag.center.astype(int) for tag in tags if tag.tag_id in self.possible_tags]
        for tag in tags:
            if tag.tag_id in self.possible_tags or tag.tag_id == 0:
                pts = tag.corners.astype(int)
                cv2.polylines(image, [pts], True, Color.GREEN.rgb, 2)
                cv2.circle(image, tag.center.astype(int), 5, Color.BLUE.rgb, -1)
                cv2.putText(
                    image, str(tag.tag_id),
                    sum_tup(get_center(*get_tag_bounds(pts)), self.tag_id_text_offset),
                    cv2.FONT_HERSHEY_PLAIN, 1, Color.GREEN.rgb
                )

        if len(tag_pts) >= 4:
            # Find the top-left and bottom-right corners of the box
            x_min, y_min, x_max, y_max = get_tag_bounds(tag_pts)

            # Draw the box with the centers of the tags
            center = get_center(x_min, y_min, x_max, y_max)
            cv2.circle(image, center, 5, Color.BLUE.rgb, -1)

            # Calculate the rotation matrix

            # TODO Karthik i got this working on hopes and dreams could u make it use a dictionary so that we dont
            #  need this stupid if statement thing.

            # TODO Done...maybe? Let me know if this is ok now

            tl_tag, tr_tag, bl_tag, br_tag = self.get_tags_at_pos(
                tags, [self.Position.TL, self.Position.TR, self.Position.BL, self.Position.BR]
            )

            if None in [tl_tag, tr_tag, bl_tag, br_tag]:
                # If missing at least 1 tag, return early since we can't compute the rect
                # should never be missing more than 1 tag
                return

            convex_hull = cv2.convexHull(np.array(tag_pts), False)
            cv2.drawContours(image, [convex_hull], -1, Color.BLUE.rgb, 3)

            return convex_hull

        return None

    # noinspection PyMethodMayBeStatic
    def contour_edges(self, image: np.ndarray) -> np.ndarray | None:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # find contours using edge biased image from canny, RETR_EXTERNAL for only outer hierarchical contours
        # CHAIN_APPROX_SIMPLE as we expect a paper to be constructed of lines, so we only need to store the 2 points
        # on the extremes of the line contour

        # see https://docs.opencv.org/4.7.0/d4/d73/tutorial_py_contours_begin.html
        # and https://docs.opencv.org/4.7.0/d9/d8b/tutorial_py_contours_hierarchy.html
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygon: np.ndarray | None = None
        if len(contours) > 0:
            # sort from greatest to the least contour area (our paper should be the biggest contour in image)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for i, contour in enumerate(sorted_contours):
                # find perimeter of contour
                contour_perimeter = cv2.arcLength(contour, True)
                # simplify the contour down to polygonal curves, using epsilon as 0.02 * perimeter
                # see https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
                # and https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
                approx_polygon: np.ndarray = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)

                # print(i, len(approx_polygon))

                # rects have 4 vertices so if this has 4 points/vertices then we assume we found paper
                if len(approx_polygon) == 4:
                    polygon = approx_polygon
                    break

            if polygon is None:
                return None

            reshaped_polygon = polygon.reshape(4, 2)
            raw_transform = four_point_transform(image, reshaped_polygon)
            grayscale_transform = four_point_transform(grayscale, reshaped_polygon)

            # self.img_window([Image.fromarray(cv2.drawContours(image, contours, -1, Color.GREEN.rgb, 3)),
            #                  Image.fromarray(raw_transform)])

            return raw_transform, grayscale_transform

        return None

    # def approximate_paper_rect(self):

    def detect_bubbles(self, grayscale: np.ndarray):
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(grayscale)

        # TODO: make like all of this better
        # this is all like unused as of right now
        ret, threshold = cv2.threshold(
            equalized_img, np.median(equalized_img), 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        blurred = cv2.GaussianBlur(equalized_img, (3, 3), 0)
        edged = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        opening = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bubble_contours = []
        copy = cv2.cvtColor(opening.copy(), cv2.COLOR_GRAY2RGB)

        for contour in contours:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)

            # print(f"x: {abs(rect_x - circle_x)}")
            # print(f"y: {abs(rect_y - circle_y)}")
            # print(f"/: {rect_w / rect_h}")
            # print(f"r: {abs((rect_w * 0.5) - radius)}")

            rect_area = rect_w * rect_h
            circle_area = math.pi * radius * radius

            # bubble_contours.append(contour)
            if (math.isclose(rect_x, circle_x, abs_tol=10)
                    and math.isclose(rect_y, circle_y, abs_tol=10)
                    and math.isclose(rect_w / rect_h, 1, abs_tol=0.5)
                    and math.isclose(rect_w * 0.5, radius, abs_tol=5)):
                if math.isclose(rect_area, 70, abs_tol=20) or (math.isclose(circle_area, 60, abs_tol=50)):
                    print(f"x: {circle_x}, y: {circle_y}, ra: {rect_area}, ca: {circle_area}")
                    cv2.rectangle(copy, (int(rect_x), int(rect_y)), (int(rect_x + rect_w), int(rect_y + rect_h)),
                                  Color.BLUE.rgb)
                    cv2.circle(copy, (int(circle_x), int(circle_y)), int(radius), Color.BLUE.rgb)
                    bubble_contours.append(contour)

        self.img_window([Image.fromarray(
            cv2.drawContours(
                copy, bubble_contours, -1, Color.GREEN.rgb
            )
        )], dims=(960, 1280))

    def update_stream(self):
        ret, frame = self.cap.read()

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect AprilTags in the frame
            self.contour_apriltags(image)

            image = Image.fromarray(image)
            image = image.resize(CameraApp.PROCESSING_DIM, Image.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.video_stream.config(image=photo)
            self.video_stream.image = photo

        self.video_stream.after(10, self.update_stream)

    def clear_preview(self):
        clear_widget(self.preview_window)

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.clear_preview()
            if self.snapshot_frame is not None:
                # self.show_preview()
                raw, grayscale = self.contour_edges(self.snapshot_frame)
                corrected = self.orientation_correction(grayscale)
                self.detect_bubbles(
                    cv2.resize(corrected, dsize=CameraApp.PROCESSING_DIM, interpolation=cv2.INTER_LANCZOS4)
                )

    def confirm_preview(self):
        self.preview_window.withdraw()

    def retry_preview(self):
        self.preview_window.withdraw()
        self.snapshot_frame = None

    def close(self):
        self.cap.release()
        # Doesn't seem like there's a way to check if window has already been destroyed
        # this isn't performance dependent so just try catch it
        try:
            self.window.destroy()
        except tk.TclError:
            pass


class FormSetupApp(CameraApp):
    JSON_PATH = "../data.json"

    PREVIEW_TARGET_FPS = 30
    _PREVIEW_TARGET_SPF = 1 / PREVIEW_TARGET_FPS

    class JsonSerializable:
        _cant_serialize = {}  # identity dict for cases where serialization can't be done
        _non_serializable_classes = [tk.BaseWidget]

        @classmethod
        def _serialize_dict(cls, obj: object):
            if any(isinstance(obj, cl) for cl in cls._non_serializable_classes):
                return cls._cant_serialize
            return obj.__dict__

        def to_json(self) -> str:
            return json.dumps(self, default=FormSetupApp.JsonSerializable._serialize_dict,
                              ensure_ascii=False, sort_keys=True, indent=4)

        def to_json_file(self, file: TextIO):
            json.dump(
                self, file, default=FormSetupApp.JsonSerializable._serialize_dict,
                ensure_ascii=False, sort_keys=True, indent=4
            )

    class Bubble(JsonSerializable):
        def __init__(self,
                     name, rect_x, rect_y, rect_w, rect_h,
                     is_tk=False, bubble_frame=None, bubble_name=None,
                     bubble_bounding_box=None, select_bubble_button=None):
            self.name = name

            self.rect_x = rect_x
            self.rect_y = rect_y
            self.rect_w = rect_w
            self.rect_h = rect_h

            self.is_tk = is_tk
            self.bubble_frame: tk.Frame | None = bubble_frame
            self.bubble_name: tk.Text | None = bubble_name
            self.bubble_bounding_box: tk.Text | None = bubble_bounding_box
            self.select_bubble_button: tk.Button | None = select_bubble_button

        def __repr__(self):
            return f"Bubble(name={self.name}," \
                   f"rect_x={self.rect_x}," \
                   f"rect_y={self.rect_y}," \
                   f"rect_w={self.rect_w}," \
                   f"rect_h={self.rect_h})"

        def _set_bounding_box(self, rect_x0, rect_y0, rect_x1, rect_y1):
            self.rect_x = rect_x0
            self.rect_y = rect_y0
            self.rect_w = (rect_x1 - rect_x0)
            self.rect_h = (rect_y1 - rect_y0)

        def _set_tk_props(self,
                          bubble_frame: tk.Frame,
                          bubble_name: tk.Text,
                          bubble_bounding_box: tk.Text,
                          select_bubble_button: tk.Button = None):
            self.is_tk = True

            self.bubble_frame = bubble_frame
            self.bubble_name = bubble_name
            self.bubble_bounding_box = bubble_bounding_box
            self.select_bubble_button = select_bubble_button

        def delete_tk_widgets(self):
            if not self.is_tk:
                raise RuntimeError("IllegalState! Cannot invoke _delete_tk_widgets when not is_tk")

            self.is_tk = False

            self.bubble_frame.destroy()
            self.bubble_name.destroy()
            self.bubble_bounding_box.destroy()

            # special case where select_bubble_button isn't always a required widget to be displayed
            if self.select_bubble_button is not None:
                self.select_bubble_button.destroy()

        def update_bounding_box(self, bounding_box: (int, int, int, int)):
            if not self.is_tk:
                raise RuntimeError("IllegalState! Cannot invoke update_bounding_box when not is_tk")

            self.bubble_bounding_box["state"] = CameraApp._TK_TEXT_NORMAL
            self.bubble_bounding_box.delete("1.0", "1.end")
            self.bubble_bounding_box.insert("1.0", str(bounding_box))
            self.bubble_bounding_box["state"] = CameraApp._TK_TEXT_DISABLED

            self._set_bounding_box(*bounding_box)

        def update_name(self):
            self.name = self.bubble_name.get("1.0", "1.end")

        @classmethod
        def _make_tk_repr(cls, parent_frame: tk.Frame, name: str, bounding_box_str: str):
            bubble_frame = tk.Frame(parent_frame)
            bubble_frame.pack(side=tk.TOP)

            bubble_name = tk.Text(bubble_frame, width=40, height=1)
            bubble_name.insert("1.0", name)
            bubble_name.pack(side=tk.TOP)

            bubble_bounding_box = tk.Text(bubble_frame, width=40, height=1)
            bubble_bounding_box.insert("1.0", bounding_box_str)
            bubble_bounding_box["state"] = FormSetupApp._TK_TEXT_NORMAL
            bubble_bounding_box.pack(side=tk.TOP)

            return bubble_frame, bubble_name, bubble_bounding_box

        def tk_repr(self,
                    container_frame: tk.Frame,
                    make_editable: bool = False,
                    selection_callback: str | Callable[[FormSetupApp.Bubble], Any] = None):
            bubble_frame, bubble_name, bubble_bounding_box = FormSetupApp.Bubble._make_tk_repr(
                container_frame,
                self.name,
                f"({self.rect_x}, {self.rect_y}, {self.rect_x + self.rect_w}, {self.rect_y + self.rect_h})"
            )

            # disable editing bubble name when making tk_repr instead of adding a new bubble by default
            bubble_name["state"] = FormSetupApp._TK_TEXT_NORMAL if make_editable else FormSetupApp._TK_TEXT_DISABLED

            select_bubble_button = None
            if make_editable and selection_callback is not None:
                select_bubble_button = tk.Button(bubble_frame, text="Select", command=lambda: selection_callback(self))
                select_bubble_button.pack(side=tk.RIGHT)

            self._set_tk_props(bubble_frame, bubble_name, bubble_bounding_box, select_bubble_button)

        @classmethod
        def from_tk(cls, settings_frame: tk.Frame, selection_callback: str | Callable[[FormSetupApp.Bubble], Any]):
            inst = cls("Enter Bubble Name Here...", 0, 0, 0, 0)

            bubble_frame, bubble_name, bubble_bounding_box = cls._make_tk_repr(
                settings_frame, inst.name, "Missing Bounding Box!"
            )

            select_bubble_button = tk.Button(bubble_frame, text="Select", command=lambda: selection_callback(inst))
            select_bubble_button.pack(side=tk.RIGHT)

            inst._set_tk_props(bubble_frame, bubble_name, bubble_bounding_box, select_bubble_button)
            return inst

        @classmethod
        def from_json(cls, input_json: dict):
            return cls(**input_json)

    class Question(JsonSerializable):
        def __init__(self, name, bubbles: [FormSetupApp.Bubble],
                     is_tk: bool = False,
                     question_frame: tk.Frame = None,
                     question_name: tk.Text = None,
                     bubbles_frame: tk.Frame = None,
                     buttons_frame: tk.Frame = None,
                     delete_button: tk.Button = None,
                     edit_button: tk.Button = None):
            self.name = name
            self.bubbles = bubbles

            self.is_tk: bool = is_tk
            self.question_frame: tk.Frame | None = question_frame
            self.question_name: tk.Text | None = question_name
            self.bubbles_frame: tk.Frame | None = bubbles_frame
            self.buttons_frame: tk.Frame | None = buttons_frame
            self.delete_button: tk.Button | None = delete_button
            self.edit_button: tk.Button | None = edit_button

        def __repr__(self):
            return f"Question(name={self.name}, bubbles={self.bubbles})"

        def _set_tk_props(self,
                          question_frame: tk.Frame,
                          question_name: tk.Text,
                          bubbles_frame: tk.Frame,
                          buttons_frame: tk.Frame,
                          delete_button: tk.Button = None,
                          edit_button: tk.Button = None):
            self.is_tk = True

            self.question_frame = question_frame
            self.question_name = question_name
            self.bubbles_frame = bubbles_frame
            self.buttons_frame = buttons_frame
            self.delete_button = delete_button
            self.edit_button = edit_button

        def delete_tk_widgets(self):
            if not self.is_tk:
                raise RuntimeError("IllegalState! Cannot invoke _delete_tk_widgets when not is_tk")

            for bubble in self.bubbles:
                bubble.delete_tk_widgets()

            self.is_tk = False

            self.question_frame.destroy()
            self.question_name.destroy()
            self.bubbles_frame.destroy()
            self.buttons_frame.destroy()

            if self.delete_button is not None:
                self.delete_button.destroy()

            if self.edit_button is not None:
                self.edit_button.destroy()

        def _delete_question(self, name: str, deletion_callback: Callable[[str], Any]):
            self.delete_tk_widgets()
            deletion_callback(name)

        def update_question(self, name: str = None, bubbles: [FormSetupApp.Bubble] = None):
            name = name if name is not None else self.name
            bubbles = bubbles if bubbles is not None else self.bubbles

            self.name = name
            self.bubbles = bubbles

            original_state = self.question_name["state"]

            # make the question name writable then change it back to its original state
            self.question_name["state"] = FormSetupApp._TK_TEXT_NORMAL
            self.question_name.delete("1.0", "1.end")
            self.question_name.insert("1.0", name)
            self.question_name["state"] = original_state

            clear_widget(self.bubbles_frame)
            for bubble in bubbles:
                bubble.tk_repr(self.bubbles_frame)

        @classmethod
        def _make_tk_repr(cls,
                          parent_frame: tk.Frame,
                          name: str,
                          bubbles: [FormSetupApp.Bubble],
                          pad_bubbles: bool = False):
            question_frame = tk.Frame(parent_frame)
            question_frame.pack(side=tk.TOP)

            question_name = tk.Text(question_frame, width=40, height=1)
            question_name.insert("1.0", name)
            question_name["state"] = FormSetupApp._TK_TEXT_DISABLED
            question_name.pack(side=tk.TOP)

            bubbles_frame = tk.Frame(question_frame)
            bubbles_frame.pack(side=tk.RIGHT)

            buttons_frame = tk.Frame(question_frame)
            if pad_bubbles:
                buttons_frame.pack(side=tk.LEFT, padx=10)
            else:
                buttons_frame.pack(side=tk.LEFT)

            for bubble in bubbles:
                bubble.tk_repr(bubbles_frame)

            return question_frame, question_name, bubbles_frame, buttons_frame

        def tk_repr(self,
                    container_frame: tk.Frame,
                    deletion_callback: str | Callable[[str], Any] = None,
                    edit_callback: str | Callable[[str, [FormSetupApp.Bubble]], Any] = None):
            edit_button, delete_button = None, None
            is_edit, is_delete = edit_callback is not None, deletion_callback is not None

            question_frame, question_name, bubbles_frame, buttons_frame = self._make_tk_repr(
                container_frame, self.name, self.bubbles, not is_edit and not is_delete
            )

            if is_edit:
                edit_button = tk.Button(
                    buttons_frame, text="Edit", command=lambda: edit_callback(self.name, self.bubbles)
                )
                edit_button.pack(side=tk.TOP)

            if is_delete:
                delete_button = tk.Button(
                    buttons_frame, text="Delete", command=lambda: self._delete_question(self.name, deletion_callback)
                )
                delete_button.pack(side=tk.BOTTOM)

            if self.is_tk:
                # make sure to clean up past tk widgets if we were already existing in the tk space
                self.delete_tk_widgets()

            self._set_tk_props(question_frame, question_name, bubbles_frame, buttons_frame, delete_button, edit_button)

        @classmethod
        def from_json(cls, input_json: dict):
            return cls(**input_json)

    class Data(JsonSerializable):
        def __init__(self, name, questions: [FormSetupApp.Question]):
            self.name = name
            self.questions = questions

        def __repr__(self):
            return f"Data(name={self.name}, questions={self.questions})"

        @classmethod
        def from_json(cls, input_json: dict):
            questions = []
            for question_dict in input_json.get("questions"):
                bubbles = [FormSetupApp.Bubble.from_json(bubble) for bubble in question_dict.get("bubbles")]
                questions.append(
                    FormSetupApp.Question(question_dict.get("name"), bubbles)
                )

            return cls(input_json.get("name"), questions)

    def __init__(self, window: tk.Tk, device: cv2.VideoCapture = None):
        super().__init__(window, device)
        try:
            with open(FormSetupApp.JSON_PATH, "r") as input_json:
                self.data = FormSetupApp.Data.from_json(json.loads(input_json.read()))
        except FileNotFoundError:
            self.data = FormSetupApp.Data("TTScout", {})
        except EnvironmentError as environ_err:
            raise RuntimeError(environ_err)

        self.setup_window = tk.Toplevel(self.window)
        self.setup_window.withdraw()

        self.question_settings_window = tk.Toplevel(self.window)
        self.question_settings_window.withdraw()

        self.canvas_container_window = tk.Toplevel(self.window)
        self.canvas_container_window.withdraw()

        self.setup_canvas = Canvas(self.canvas_container_window)
        self.canvas_preview_update = False  # continue updating while this is true, stop when false
        self.canvas_preview_update_thread: threading.Thread | None = None

        self.setup_image: np.ndarray | None = None
        self.settings_frame: tk.Frame | None = None
        self.shown_bubbles_scroll_frame: ScrollFrame | None = None

        self.shown_questions_container: tk.Frame | None = None
        self.shown_questions_scroll_frame: ScrollFrame | None = None
        self.curr_question_setting_bubbles = []

    def display_question(self, question: Question):
        if self.shown_questions_scroll_frame is not None:
            question.tk_repr(
                self.shown_questions_scroll_frame.viewPort,
                lambda n: self.remove_question(n),
                lambda n, b: self.make_question_settings_window(n, b, is_update=True)
            )

    def update_question(self, question: Question, name: str, bubbles: [Bubble]):
        if self.shown_questions_scroll_frame is not None:
            question.update_question(name, bubbles)

    def add_question(self, question: str, bubbles: [Bubble]):
        existing_question = self.get_question(question)
        if existing_question is None:
            new_question = FormSetupApp.Question(question, bubbles)
            self.data.questions.append(new_question)
            self.display_question(new_question)

    def add_question_direct(self, question: Question):
        existing_question = self.get_question(question.name)
        if existing_question is None:
            self.data.questions.append(question)
            self.display_question(question)

    def get_question(self, question_name: str) -> [Bubble]:
        return next((question for question in self.data.questions if question_name == question.name), None)

    def remove_question(self, question_name: str):
        existing_question = self.get_question(question_name)
        if existing_question is not None:
            self.data.questions.remove(existing_question)
            # no need to do any removal of UI/Widgets here as the question should delete its own widgets

    def make_setup_window(self, setup_image: Image, dims=(240, 320)):
        self.setup_window.protocol("WM_DELETE_WINDOW", self.close_setup_window)

        # Create a PhotoImage object for the setup_image
        img = setup_image.resize(dims, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        # Display the image in a vertical stack
        if self.setup_window.winfo_exists():
            self.setup_window.deiconify()

        setup_frame = tk.Frame(self.setup_window)
        setup_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.shown_questions_container = tk.Frame(setup_frame)
        self.shown_questions_container.pack(side=tk.TOP)

        self.shown_questions_scroll_frame = ScrollFrame(self.shown_questions_container)
        self.shown_questions_scroll_frame.pack(side=tk.TOP, fill="both", expand=True)

        for question in self.data.questions:
            self.display_question(question)

        setup_label = tk.Label(setup_frame, image=photo)
        setup_label.image = photo  # Keep reference to avoid garbage collection
        setup_label.pack(side=tk.TOP)

        add_question_button = tk.Button(self.setup_window, text="Add Question", command=self.add_question_btn_callback)
        add_question_button.pack(side=tk.LEFT)

        save_exit_button = tk.Button(self.setup_window, text="Save & Exit", command=self.close_setup_window)
        save_exit_button.pack(side=tk.RIGHT)

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.clear_preview()
            if self.snapshot_frame is not None:
                raw_transform, grayscale_transform = self.contour_edges(self.snapshot_frame)
                corrected = self.orientation_correction(grayscale_transform)
                processed = self.process_snapshot(corrected)

                self.setup_image = processed
                self.img_window([Image.fromarray(corrected)], dims=CameraApp.PROCESSING_DIM)

    def confirm_preview(self):
        self.preview_window.withdraw()
        self.clear_preview()

        self.snapshot_frame = None
        # self.img_window([Image.fromarray(self.setup_frame)], dims=(640, 480))
        self.make_setup_window(Image.fromarray(self.setup_image), dims=(320, 240))

    def close_setup_window(self):
        self.setup_window.withdraw()
        clear_widget(self.setup_window)
        self.setup_image = None

    def add_question_btn_callback(self):
        if self.settings_frame is not None:
            self.close_question_settings_window_callback()

        self.make_question_settings_window()

    def make_question_settings_window(
            self,
            name: str = "Enter Question Name Here...",
            bubbles: [Bubble] = None,
            is_update: bool = False):
        self.question_settings_window.protocol("WM_DELETE_WINDOW", self.close_question_settings_window_callback)

        # Deiconify/Show the window
        if self.question_settings_window.winfo_exists():
            self.question_settings_window.deiconify()

        self.settings_frame = tk.Frame(self.question_settings_window)
        self.settings_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.shown_bubbles_scroll_frame = ScrollFrame(self.settings_frame)
        self.shown_bubbles_scroll_frame.pack(side=tk.BOTTOM, fill="both", expand=True)

        question_name = tk.Text(self.settings_frame, width=40, height=1)
        question_name.insert("1.0", name)
        question_name.pack(side=tk.TOP)

        add_bubble_button = tk.Button(self.settings_frame, text="Add Bubble", command=self.add_bubble_btn_callback)
        add_bubble_button.pack(side=tk.LEFT)

        if bubbles is not None:
            for bubble in bubbles:
                self.curr_question_setting_bubbles.append(bubble)
                bubble.tk_repr(
                    self.shown_bubbles_scroll_frame.viewPort,
                    make_editable=True,
                    selection_callback=lambda b=bubble: self.make_canvas_window(
                        Image.fromarray(self.setup_image),
                        b, dims=CameraApp.PROCESSING_DIM
                    )
                )

        confirm_button = tk.Button(
            self.question_settings_window,
            text="Confirm",
            command=lambda: self.close_question_settings_window_callback(
                question_name.get("1.0", "1.end"), self.curr_question_setting_bubbles, is_update, name
            )
        )
        confirm_button.pack(side=tk.RIGHT)

        cancel_button = tk.Button(
            self.question_settings_window, text="Cancel", command=self.close_question_settings_window_callback
        )
        cancel_button.pack(side=tk.LEFT)

    def add_bubble_btn_callback(self):
        bubble = FormSetupApp.Bubble.from_tk(self.shown_bubbles_scroll_frame.viewPort, self.select_bubble_btn_callback)
        self.curr_question_setting_bubbles.append(bubble)

    def close_question_settings_window_callback(
            self,
            question_name: str = None,
            question_bubbles: [Bubble] = None,
            is_update: bool = False,
            previous_name: str = None):
        if question_name is not None and question_bubbles is not None:
            copied_bubbles = question_bubbles.copy()  # need to make a copy here, so it doesn't get gc 'ed early
            for bubble in copied_bubbles:
                bubble.update_name()

            if not is_update:
                question = FormSetupApp.Question(question_name, copied_bubbles)
                self.add_question_direct(question)
            else:
                question = self.get_question(previous_name)
                self.update_question(question, question_name, copied_bubbles)

        self.curr_question_setting_bubbles.clear()

        self.question_settings_window.withdraw()
        clear_widget(self.question_settings_window)

    def select_bubble_btn_callback(self, bubble: Bubble):
        self.make_canvas_window(Image.fromarray(self.setup_image), bubble, dims=CameraApp.PROCESSING_DIM)

    def make_canvas_window(self, canvas_image: Image, bubble: Bubble, dims=(240, 320)):
        self.canvas_container_window.protocol("WM_DELETE_WINDOW", self.close_canvas_window_callback)

        # Deiconify/Show the window
        if self.canvas_container_window.winfo_exists():
            self.canvas_container_window.deiconify()

        # Create a PhotoImage object for the setup_image
        img = canvas_image.resize(dims, Image.LANCZOS)
        img_mat = np.array(img)
        photo = ImageTk.PhotoImage(img)
        photo_width, photo_height = photo.width(), photo.height()

        def crop(crop_img: Image, rect: (int, int, int, int), post_resize=(160, 160)) -> Image:
            return crop_img.crop(rect).resize(post_resize, Image.LANCZOS)

        preview_frame = tk.Frame(self.canvas_container_window)
        preview_frame.pack(side=tk.TOP, padx=10, pady=10)

        preview_label = tk.Label(preview_frame, image=photo)
        preview_label.image = photo  # Keep reference to avoid garbage collection
        preview_label.pack(side=tk.TOP)

        preview_text = tk.Text(preview_frame, width=40, height=1)
        preview_text.insert("1.0", "No Data Yet...")
        preview_text["state"] = FormSetupApp._TK_TEXT_DISABLED
        preview_text.pack(side=tk.LEFT)

        rect_text = tk.Text(preview_frame, width=40, height=1)
        rect_text.insert("1.0", "No Data Yet...")
        rect_text["state"] = FormSetupApp._TK_TEXT_DISABLED
        rect_text.pack(side=tk.LEFT)

        _baseline_percent = 100 * CameraApp.BUBBLE_DETECTION_BASELINE

        def update_preview():
            while self.canvas_preview_update:
                cropped_img = crop(img, self.setup_canvas.rect_absolute)

                tk_img = ImageTk.PhotoImage(cropped_img)
                if preview_label.winfo_exists() == 1:
                    preview_label.config(image=tk_img)
                    preview_label.image = tk_img

                rect_x0, rect_y0, rect_x1, rect_y1 = self.setup_canvas.rect_absolute
                extracted_img: np.ndarray = img_mat[rect_y0:rect_y1, rect_x0:rect_x1]

                n_px = (rect_x1 - rect_x0) * (rect_y1 - rect_y0)
                n_white_px = np.count_nonzero(extracted_img)
                prop_white = n_white_px / n_px

                if preview_text.winfo_exists() == 1:
                    preview_text["state"] = FormSetupApp._TK_TEXT_NORMAL
                    preview_text.delete("1.0", "1.end")
                    preview_text.insert("1.0", f"{(100 * prop_white):3.3f}% white; baseline={_baseline_percent}")
                    preview_text["state"] = FormSetupApp._TK_TEXT_DISABLED

                if rect_text.winfo_exists() == 1:
                    rect_text["state"] = FormSetupApp._TK_TEXT_NORMAL
                    rect_text.delete("1.0", "1.end")
                    rect_text.insert("1.0", f"({rect_x0}, {rect_y0}, {rect_x1}, {rect_y1})")
                    rect_text["state"] = FormSetupApp._TK_TEXT_DISABLED

                time.sleep(FormSetupApp._PREVIEW_TARGET_SPF)

        self.canvas_preview_update = True
        self.canvas_preview_update_thread = threading.Thread(target=update_preview)
        self.canvas_preview_update_thread.start()

        self.setup_canvas.create_image(0.5 * photo_width, 0.5 * photo_height, image=photo)
        self.setup_canvas.image = photo  # Keep reference to avoid garbage collection

        if bubble.rect_w != 0 and bubble.rect_h != 0:
            self.setup_canvas.set_rect_pos(
                (bubble.rect_x, bubble.rect_y, bubble.rect_x + bubble.rect_w, bubble.rect_y + bubble.rect_h)
            )

        self.setup_canvas.config(width=photo_width, height=photo_height)

        # noinspection PyProtectedMember
        self.setup_canvas.tag_raise(self.setup_canvas._tag)
        self.setup_canvas.pack(fill="both", expand=True)

        confirm_button = tk.Button(
            self.canvas_container_window, text="Confirm", command=lambda: self.confirm_canvas_callback(bubble)
        )
        confirm_button.pack(side=tk.LEFT)

    def close_canvas_window_callback(self):
        self.canvas_container_window.withdraw()
        clear_widget(self.canvas_container_window, exclude=self.setup_canvas)

    def confirm_canvas_callback(self, bubble: Bubble):
        bubble.update_bounding_box(self.setup_canvas.rect_absolute)

        self.canvas_preview_update = False
        self.close_canvas_window_callback()

    def close(self):
        try:
            with open(FormSetupApp.JSON_PATH, "w", encoding="utf-8") as output_json:
                self.data.to_json_file(output_json)
        except EnvironmentError as environ_err:
            raise RuntimeError(environ_err)

        super().close()


class CaptureResponsesApp(CameraApp):
    CENTER_COMPLETENESS_THRESHOLD = 0.8
    CENTER_COMPLETENESS_WIDE_BLANK_THRESHOLD = 0.2

    MONO_REGION_ALTERNATIVE_AREA_THRESHOLD = 25

    RESULT_DISPLAY_DIM = (640, 480)

    OUTPUT_FILE_PATH = "../output.csv"

    INTERACTION_BOUND_ACTION = "<Button-1>"

    class QuestionDetection:
        def __init__(self, question: FormSetupApp.Question, detections: list[bool]):
            self.question = question
            self.detections = detections

            self.bubbles = dict(zip(question.bubbles, detections))
            self.tk_container: tk.Frame | None = None

            self.detection_result_texts: dict[FormSetupApp.Bubble, tk.Text] | None = None

        def __repr__(self):
            return f"QuestionDetection(" \
                   f"question={self.question}," \
                   f"detections={self.detections})"

        def tk_repr(self, parent_frame: tk.Frame):
            primary_container = tk.Frame(parent_frame)

            question_container = tk.Frame(primary_container)
            question_container.pack(side=tk.LEFT)

            result_container = tk.Frame(primary_container)
            result_container.pack(side=tk.RIGHT)

            detection_result_texts = {}
            for bubble, detection in self.bubbles.items():
                detection_result_text = tk.Text(result_container, width=40, height=1)
                detection_result_text.insert("1.0", "Yes" if detection else "No")
                detection_result_text["state"] = CaptureResponsesApp._TK_TEXT_DISABLED
                detection_result_text.pack(side=tk.TOP, pady=10)

                detection_result_texts[bubble] = detection_result_text

            primary_container.pack(side=tk.TOP)

            self.question.tk_repr(question_container)
            self.tk_container = primary_container
            self.detection_result_texts = detection_result_texts

        def update_detection(self, updated_detections: dict[FormSetupApp.Bubble, bool] = None):
            self.bubbles.update(updated_detections)

            for bubble, detection in self.bubbles.items():
                result_text = self.detection_result_texts.get(bubble, None)
                if result_text is not None:
                    original_state = result_text["state"]
                    result_text["state"] = CaptureResponsesApp._TK_TEXT_NORMAL
                    result_text.delete("1.0", "1.end")
                    result_text.insert("1.0", "Yes" if detection else "No")
                    result_text["state"] = original_state

    class FieldPosition(enum.Enum):
        BlueOne = "Blue 1"
        BlueTwo = "Blue 2"
        BlueThree = "Blue 3"

        RedOne = "Red 1"
        RedTwo = "Red 2"
        RedThree = "Red 3"

    class MatchType(enum.Enum):
        Qualification = "Qualification"

        # Double elimination bracket
        # https://www.firstinspires.org/sites/default/files/uploads/resource_library/frc/game-and-season-info/competition-manual
        # double_elimination_playoff_communication.pdf
        # double_elimination_bracket.jpg

        PlayoffsRoundOne = "Playoffs Round 1"
        PlayoffsRoundTwo = "Playoffs Round 2"
        PlayoffsRoundThree = "Playoffs Round 3"
        PlayoffsRoundFour = "Playoffs Round 4"
        PlayoffsRoundFive = "Playoffs Round 5"

        PlayoffsFinal = "Playoffs Final"

        @classmethod
        def get_playoffs_type(cls, playoffs_match_n: int) -> CaptureResponsesApp.MatchType:
            # match 14-16 treated as finals match (best 2 out of 3)
            if not (1 <= playoffs_match_n <= 16):
                raise ValueError("playoffs_match_n must be between 1 and 14, inclusive")

            if 1 <= playoffs_match_n <= 4:
                return cls.PlayoffsRoundOne
            elif 5 <= playoffs_match_n <= 8:
                return cls.PlayoffsRoundTwo
            elif 9 <= playoffs_match_n <= 10:
                return cls.PlayoffsRoundThree
            elif 11 <= playoffs_match_n <= 12:
                return cls.PlayoffsRoundFour
            elif playoffs_match_n == 13:
                return cls.PlayoffsRoundFive
            else:
                return cls.PlayoffsFinal

    # TODO: decide if the MatchType field actually needs to be implemented (mainly just for playoffs scouting)
    class OutputHeaders(enum.Enum):
        Match = "Match #"
        # MatchType = "MatchType"
        Team = "Team #"
        Position = "Position"
        Scouter = "Scouter"
        Question = "Question"
        Bubble = "Bubble"
        Value = "Value"

        @classmethod
        def get_headers(cls):
            return [e.value for e in cls]

    class Output:

        def __init__(self, output_file_path: str):
            self.output_file_path = output_file_path
            self.headers = CaptureResponsesApp.OutputHeaders.get_headers()
            self.data = []

        def give_detection(self,
                           match: int,
                           team: int,
                           position: CaptureResponsesApp.FieldPosition,
                           scouter: str,
                           detections: list[CaptureResponsesApp.QuestionDetection]):

            output_headers = CaptureResponsesApp.OutputHeaders
            for detection in detections:
                for bubble, detected in detection.bubbles.items():
                    self.data.append({
                        output_headers.Match.value: str(match),
                        output_headers.Team.value: str(team),
                        output_headers.Position.value: position.value,
                        output_headers.Scouter.value: scouter,
                        output_headers.Question.value: detection.question.name,
                        output_headers.Bubble.value: bubble.name,
                        output_headers.Value.value: detected
                    })

        def write(self):
            if len(self.data) <= 0:
                # don't write anything if we don't have any data, no need to just write headers
                return

            already_exists = os.path.isfile(self.output_file_path)

            with open(self.output_file_path, "w" if not already_exists else "a", newline="") as output_file:
                csv_writer = csv.DictWriter(output_file, fieldnames=self.headers)
                if not already_exists:
                    csv_writer.writeheader()

                for row in self.data:
                    csv_writer.writerow(row)

    def __init__(self, window: tk.Tk, device: cv2.VideoCapture = None):
        super().__init__(window, device)
        try:
            with open(FormSetupApp.JSON_PATH, "r") as input_json:
                self.data = FormSetupApp.Data.from_json(json.loads(input_json.read()))
        except FileNotFoundError:
            raise RuntimeError("Cannot start CaptureResponsesApp when data.json does not exist!")
        except EnvironmentError as environ_err:
            raise RuntimeError(environ_err)

        self.responses_window = tk.Toplevel(self.window)
        self.responses_window.withdraw()

        self.responses_container: tk.Frame | None = None
        self.responses_image_container: tk.Frame | None = None

        self.held_detections: list[CaptureResponsesApp.QuestionDetection] | None = None

        self.output = CaptureResponsesApp.Output(CaptureResponsesApp.OUTPUT_FILE_PATH)

        self.scouter_name_text: tk.Text | None = None
        self.match_number_text: tk.Text | None = None
        self.match_type_dropdown_var: tk.StringVar | None = None
        self.team_number_text: tk.Text | None = None
        self.position_dropdown_var: tk.StringVar | None = None

        self.held_settings: dict[CaptureResponsesApp.OutputHeaders, str] | None = None
        self.held_detection_interaction_binding: str | None = None

        self.held_detection_threshold: np.ndarray | None = None
        self.detection_img_label: tk.Label | None = None

        # detection: QuestionDetection, bubble: Bubble, detected: bool
        self.update_detection_arg_map: \
            dict[
                FormSetupApp.Bubble,
                tuple[
                    CaptureResponsesApp.QuestionDetection,
                    list[CaptureResponsesApp.QuestionDetection],
                    FormSetupApp.Bubble,
                    bool
                ]
            ] | None = None

        self.update_detection_rect_map: dict[FormSetupApp.Bubble, tuple[int, int, int, int]] | None = None

    @classmethod
    def check_bubble_center_completeness(cls, bubble: np.ndarray):
        row_regions = []
        widest_blanks = []
        for row in bubble:
            regions = consecutive(row, step_size=0)
            blank_regions = [region for region in regions if all(n == 0 for n in region)]
            widest_blank = max(map(len, blank_regions)) if len(blank_regions) > 0 else 0

            row_regions.append(regions)
            widest_blanks.append(widest_blank)

        center_complete = sum(len(region) <= 3 for region in row_regions)
        total_rows, total_cols = bubble.shape

        average_widest_blank = np.mean(widest_blanks)

        return ((center_complete / total_rows) >= CaptureResponsesApp.CENTER_COMPLETENESS_THRESHOLD) \
            or ((average_widest_blank / total_cols) <= CaptureResponsesApp.CENTER_COMPLETENESS_WIDE_BLANK_THRESHOLD)

    @classmethod
    def check_bubble_mono_contour(cls, bubble: np.ndarray):
        contours, hierarchy = cv2.findContours(bubble, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return len(hierarchy) == 1 if hierarchy is not None else True

    @classmethod
    def check_bubble_white_px_prop(cls, width: int, height: int, bubble: np.ndarray):
        n_px, n_white_px = width * height, np.count_nonzero(bubble)
        proportion = n_white_px / n_px

        return proportion >= CaptureResponsesApp.BUBBLE_DETECTION_BASELINE

    @classmethod
    def check_bubble_mono_region(cls, bubble: np.ndarray):
        labeled, n_labels = ndi.label(bubble)
        if n_labels <= 1:
            return True

        region_areas = []
        for i in range(1, n_labels + 1):
            region_areas.append(len(labeled[labeled == i]))

        threshold = CaptureResponsesApp.MONO_REGION_ALTERNATIVE_AREA_THRESHOLD

        max_area = max(region_areas)
        primary = max_area > threshold
        remaining = all(area <= threshold for area in region_areas if area != max_area)

        return primary and remaining

    def detect_bubbles(self, grayscale: np.ndarray):
        threshold = self.process_snapshot(grayscale)
        # disp_copy = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2RGB)

        marked_bubbles_by_question = []
        for question in self.data.questions:
            marked_bubbles = []

            for bubble in question.bubbles:
                rect_x0, rect_y0 = bubble.rect_x, bubble.rect_y
                rect_x1, rect_y1 = rect_x0 + bubble.rect_w, rect_y0 + bubble.rect_h
                extracted_img: np.ndarray = threshold[rect_y0:rect_y1, rect_x0:rect_x1]

                # cv2.rectangle(disp_copy, (rect_x0, rect_y0), (rect_x1, rect_y1), Color.GREEN.rgb)
                # cv2.circle(copy, (int(circle_x), int(circle_y)), int(radius), Color.RED.rgb)

                # flat_nonzero = np.flatnonzero(extracted_img) % bubble.rect_w
                # if len(flat_nonzero) > 0:
                #     leftmost_y = np.min(flat_nonzero)
                #     rightmost_y = np.max(flat_nonzero)
                #
                #     print(f"ly: {leftmost_y}, ry: {rightmost_y}")

                center_completeness = CaptureResponsesApp.check_bubble_center_completeness(extracted_img)
                mono_contour = CaptureResponsesApp.check_bubble_mono_contour(extracted_img)
                white_px_prop_above_baseline = CaptureResponsesApp.check_bubble_white_px_prop(
                    bubble.rect_w, bubble.rect_h, extracted_img
                )
                mono_region = CaptureResponsesApp.check_bubble_mono_region(extracted_img)

                conditions = [center_completeness, mono_contour, white_px_prop_above_baseline, mono_region]
                meets_all_conditions = all(conditions)

                # print(extracted_img)
                # print(f"cc: {center_completeness}, "
                #       f"mc: {mono_contour}, "
                #       f"wp: {white_px_prop_above_baseline}, "
                #       f"mr: {mono_region}")
                #
                # print(f"{question.name}:{bubble.name}="
                #       f"{meets_all_conditions}")

                # if meets_all_conditions:
                #     cv2.rectangle(disp_copy, (rect_x0, rect_y0), (rect_x1, rect_y1), Color.GREEN.rgb)
                # else:
                #     cv2.rectangle(disp_copy, (rect_x0, rect_y0), (rect_x1, rect_y1), Color.RED.rgb)

                marked_bubbles.append(meets_all_conditions)

            marked_bubbles_by_question.append(marked_bubbles)

        # self.img_window([Image.fromarray(disp_copy)], dims=CameraApp.PROCESSING_DIM)
        # self.img_window([Image.fromarray(shadow_filtered_image)], dims=CameraApp.PROCESSING_DIM)

        return threshold, self.data.questions, marked_bubbles_by_question

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.clear_preview()
            if self.snapshot_frame is not None:
                raw, grayscale = self.contour_edges(self.snapshot_frame)
                corrected = self.orientation_correction(grayscale)

                threshold, questions, marked_bubbles = self.detect_bubbles(
                    cv2.resize(corrected, dsize=CameraApp.PROCESSING_DIM, interpolation=cv2.INTER_LANCZOS4)
                )

                self.process_detection_results(threshold, raw, questions, marked_bubbles)

    def close_displayed_detection(self):
        self.responses_window.withdraw()

        if self.held_detections is not None:
            for detection in self.held_detections:
                detection.question.delete_tk_widgets()

            self.held_detections = None

        if self.held_detection_interaction_binding is not None:
            self.detection_img_label.unbind(
                CaptureResponsesApp.INTERACTION_BOUND_ACTION, self.held_detection_interaction_binding
            )

            self.held_detection_interaction_binding = None

        self.held_settings = {
            CaptureResponsesApp.OutputHeaders.Scouter: self.scouter_name_text.get("1.0", "1.end"),
            CaptureResponsesApp.OutputHeaders.Match: self.match_number_text.get("1.0", "1.end"),
            # CaptureResponsesApp.OutputHeaders.MatchType: self.match_type_dropdown_var.get(),
            CaptureResponsesApp.OutputHeaders.Team: self.team_number_text.get("1.0", "1.end"),
            CaptureResponsesApp.OutputHeaders.Position: self.position_dropdown_var.get(),
        }

        clear_widget(self.responses_window)

        self.held_detection_threshold = None
        self.detection_img_label = None

        self.responses_image_container = None
        self.scouter_name_text = None
        self.match_number_text = None
        self.match_type_dropdown_var = None
        self.team_number_text = None
        self.position_dropdown_var = None

    @classmethod
    def make_text_entry(cls, parent_frame: tk.Frame, label_name: str, default_value: str):
        label_text = tk.Text(parent_frame, width=10, height=1)
        label_text.insert("1.0", label_name)
        label_text["state"] = CaptureResponsesApp._TK_TEXT_DISABLED

        text = tk.Text(parent_frame, width=10, height=1)
        text.insert("1.0", default_value)

        return label_text, text

    @classmethod
    def make_dropdown_entry(cls,
                            detection_setup_container: tk.Frame,
                            enum_cls: enum.Enum.__class__,
                            name: str,
                            default_val: str):
        frame = tk.Frame(detection_setup_container)
        str_var = tk.StringVar(frame)
        str_var.set(default_val)

        label_text = tk.Text(frame, width=10, height=1)
        label_text.insert("1.0", name)
        label_text["state"] = CaptureResponsesApp._TK_TEXT_DISABLED

        dropdown = tk.OptionMenu(
            frame, str_var,
            *[m_type.name for m_type in enum_cls if m_type != default_val]
        )

        return frame, str_var, label_text, dropdown

    @classmethod
    def pack_detection_entry(cls, label_text: tk.Widget, text: tk.Widget, frame: tk.Frame):
        label_text.pack(side=tk.LEFT)
        text.pack(side=tk.RIGHT)
        frame.pack(side=tk.TOP)

    def update_detection(self,
                         detection: CaptureResponsesApp.QuestionDetection,
                         detections: list[CaptureResponsesApp.QuestionDetection],
                         bubble: FormSetupApp.Bubble,
                         detected: bool):
        if self.held_detection_threshold is not None:
            threshold_copy = cv2.cvtColor(self.held_detection_threshold.copy(), cv2.COLOR_GRAY2RGB)

            for next_detection in detections:
                for next_bubble, next_detected in next_detection.bubbles.items():
                    color_detect_val = next_detected if next_bubble != bubble else detected
                    cv2.rectangle(
                        threshold_copy,
                        (next_bubble.rect_x, next_bubble.rect_y),
                        (next_bubble.rect_x + next_bubble.rect_w, next_bubble.rect_y + next_bubble.rect_h),
                        (Color.GREEN.rgb if color_detect_val else Color.RED.rgb)
                    )

            threshold_image = Image.fromarray(threshold_copy)
            resized_threshold = threshold_image.resize(CaptureResponsesApp.RESULT_DISPLAY_DIM, Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_threshold)

            if self.detection_img_label is not None:
                self.detection_img_label.config(image=photo)
                self.detection_img_label.image = photo

        self.update_detection_arg_map[bubble] = (detection, detections, bubble, not detected)
        detection.update_detection({bubble: detected})

    def display_detection_image(self, threshold: np.ndarray, detections: list[CaptureResponsesApp.QuestionDetection]):
        self.responses_window.protocol("WM_DELETE_WINDOW", self.close_displayed_detection)

        self.responses_image_container = tk.Frame(self.responses_window)
        self.responses_image_container.pack(side=tk.LEFT, padx=10, pady=10)

        detection_setup_container = tk.Frame(self.responses_window)
        detection_text_container = ScrollFrame(self.responses_window, width=480, height=480)

        if self.held_settings is not None:
            scouter = self.held_settings.get(CaptureResponsesApp.OutputHeaders.Scouter)
            match_number = self.held_settings.get(CaptureResponsesApp.OutputHeaders.Match)
            # match_type = self.held_settings.get(CaptureResponsesApp.OutputHeaders.MatchType)
            position = self.held_settings.get(CaptureResponsesApp.OutputHeaders.Position)
            team = self.held_settings.get(CaptureResponsesApp.OutputHeaders.Team)
        else:
            scouter = "Harry"
            match_number = "0"
            # _random_match_type: CaptureResponsesApp.MatchType = random.choice(
            #     list(CaptureResponsesApp.MatchType)
            # )

            _random_field_position: CaptureResponsesApp.FieldPosition = random.choice(
                list(CaptureResponsesApp.FieldPosition)
            )

            # match_type = _random_match_type.name
            position = _random_field_position.name
            team = "1683"

        # Scouter
        scouter_name_frame = tk.Frame(detection_setup_container)
        scouter_name_label_text, scouter_name_text = CaptureResponsesApp.make_text_entry(
            scouter_name_frame, CaptureResponsesApp.OutputHeaders.Scouter.value, scouter
        )

        # Match #
        match_number_frame = tk.Frame(detection_setup_container)
        match_number_label_text, match_number_text = CaptureResponsesApp.make_text_entry(
            match_number_frame, CaptureResponsesApp.OutputHeaders.Match.value, match_number
        )

        # Team #
        team_number_frame = tk.Frame(detection_setup_container)
        team_number_label_text, team_number_text = CaptureResponsesApp.make_text_entry(
            team_number_frame, CaptureResponsesApp.OutputHeaders.Team.value, team
        )

        # Scouter
        CaptureResponsesApp.pack_detection_entry(scouter_name_label_text, scouter_name_text, scouter_name_frame)

        # Match #
        CaptureResponsesApp.pack_detection_entry(match_number_label_text, match_number_text, match_number_frame)

        # MatchType
        # match_type_frame, match_type_str_var, match_type_label_text, match_type_dropdown = \
        #     CaptureResponsesApp.make_dropdown_entry(
        #         detection_setup_container,
        #         CaptureResponsesApp.MatchType,
        #         CaptureResponsesApp.OutputHeaders.MatchType.value,
        #         match_type
        #     )
        #
        # CaptureResponsesApp.pack_detection_entry(match_type_label_text, match_type_dropdown, match_type_frame)

        # Team #
        CaptureResponsesApp.pack_detection_entry(team_number_label_text, team_number_text, team_number_frame)

        # Position
        position_frame, position_str_var, position_label_text, position_dropdown = \
            CaptureResponsesApp.make_dropdown_entry(
                detection_setup_container,
                CaptureResponsesApp.FieldPosition,
                CaptureResponsesApp.OutputHeaders.Position.value,
                position
            )

        CaptureResponsesApp.pack_detection_entry(position_label_text, position_dropdown, position_frame)

        detection_setup_container.pack(side=tk.TOP)

        self.scouter_name_text = scouter_name_text
        self.match_number_text = match_number_text
        # self.match_type_dropdown_var = match_type_str_var
        self.team_number_text = team_number_text
        self.position_dropdown_var = position_str_var

        threshold_copy = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2RGB)
        self.held_detection_threshold = threshold

        if self.update_detection_arg_map is None:
            self.update_detection_arg_map = {}

        if self.update_detection_rect_map is None:
            self.update_detection_rect_map = {}

        for detection in detections:
            detection.tk_repr(detection_text_container.viewPort)

            for bubble, detected in detection.bubbles.items():
                self.update_detection_arg_map[bubble] = (detection, detections, bubble, not detected)
                self.update_detection_rect_map[bubble] = \
                    (bubble.rect_x, bubble.rect_y, bubble.rect_x + bubble.rect_w, bubble.rect_y + bubble.rect_h)

                cv2.rectangle(
                    threshold_copy,
                    (bubble.rect_x, bubble.rect_y),
                    (bubble.rect_x + bubble.rect_w, bubble.rect_y + bubble.rect_h),
                    Color.GREEN.rgb if detected else Color.RED.rgb
                )

        detection_text_container.pack(side=tk.TOP)

        threshold_image = Image.fromarray(threshold_copy)
        resized_threshold = threshold_image.resize(CaptureResponsesApp.RESULT_DISPLAY_DIM, Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized_threshold)

        if self.responses_window.winfo_exists():
            self.responses_window.deiconify()

        image_label = tk.Label(self.responses_image_container, image=photo)
        image_label.image = photo  # Keep reference to avoid garbage collection
        image_label.pack(side=tk.TOP)

        # setup bindings for manually updating/toggling detections
        # must be done *after* image_label is created or else I'd have to bind it to the responses_image_container
        # which might not always work?
        def on_image_label_interaction(event: tk.Event):
            for next_bubble, bounding_rect in self.update_detection_rect_map.items():
                if pt_inside_rect(*bounding_rect, event.x, event.y):
                    self.update_detection(*self.update_detection_arg_map[next_bubble])

        interaction_bound_name = image_label.bind(
            CaptureResponsesApp.INTERACTION_BOUND_ACTION, on_image_label_interaction
        )

        self.held_detection_interaction_binding = interaction_bound_name
        self.detection_img_label = image_label

        confirm_button = tk.Button(self.responses_image_container, text="Confirm", command=self.confirm_preview)
        confirm_button.pack(side=tk.LEFT)

        retry_button = tk.Button(self.responses_image_container, text="Retry", command=self.retry_preview)
        retry_button.pack(side=tk.RIGHT)

    def confirm_preview(self):
        if self.held_detections is not None:
            self.output.give_detection(
                match=int(self.match_number_text.get("1.0", "1.end")),
                team=int(self.team_number_text.get("1.0", "1.end")),
                position=CaptureResponsesApp.FieldPosition[self.position_dropdown_var.get()],
                scouter=self.scouter_name_text.get("1.0", "1.end"),
                detections=self.held_detections
            )
        else:
            warnings.warn("self.held_detections was None when attempting to save data!")

        # make sure to only close AFTER we call give_detection that way self.held_detections doesn't get set to None
        self.close_displayed_detection()

    def retry_preview(self):
        self.close_displayed_detection()

    def process_detection_results(self,
                                  threshold: np.ndarray,
                                  raw: np.ndarray,
                                  questions: [FormSetupApp.Question],
                                  detections: list[list[bool]]):
        question_detections: [CaptureResponsesApp.QuestionDetection] = [
            CaptureResponsesApp.QuestionDetection(questions[i], detections[i]) for i in range(len(questions))
        ]

        if self.held_detections is not None:
            raise RuntimeError("cannot start another process_detection_results when one is already held!")

        self.held_detections = question_detections.copy()
        self.display_detection_image(threshold, question_detections)

    def close(self):
        super().close()
        self.output.write()


class SelectionMenuApp:
    PREVIEW_STREAM_DIM = (320, 240)
    MAX_SUPPORTED_CAPTURE_DEVICES = 12

    class Mode(enum.Enum):
        FormSetup = 1
        CaptureResponses = 2
        Camera = 3

    def __init__(self, window: tk.Tk):
        self.window = window
        self.window.title("SelectionMenu")

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.mode_selection_window = tk.Toplevel(self.window)
        self.mode_selection_window.withdraw()

        self.selected_camera_id = None
        self.selected_camera = None

        self.selected_mode = None

        self.instance: CameraApp | FormSetupApp | CaptureResponsesApp | None = None

        self.capture_devices = []
        for capture_id in range(SelectionMenuApp.MAX_SUPPORTED_CAPTURE_DEVICES):
            device = get_capture_device(capture_id, suppress_warn=True)
            if device is None:
                break

            self.capture_devices.append(device)

        self.video_stream_labels: [tk.Label] = []
        self.select_camera_buttons: [tk.Button] = []

        for i in range(len(self.capture_devices)):
            stream_label = tk.Label(self.video_frame)
            stream_label.pack(side=tk.TOP)

            select_camera_button = tk.Button(self.video_frame, text="Select", command=lambda: self.select_camera(i))
            select_camera_button.pack(side=tk.RIGHT, padx=10, pady=10)

            self.select_camera_buttons.append(select_camera_button)
            self.video_stream_labels.append(stream_label)

    def select_camera(self, device_id: int):
        self.selected_camera_id = device_id
        self.selected_camera = self.capture_devices[device_id]

        self.make_mode_selection_window()
        self.close_camera_selection_window()

    def close_camera_selection_window(self):
        clear_widget(self.video_frame)

    def make_mode_selection_window(self):
        # self.mode_selection_window.protocol("WM_DELETE_WINDOW", self.close_canvas_window_callback)

        # Deiconify/Show the window
        if self.mode_selection_window.winfo_exists():
            self.mode_selection_window.deiconify()

        selection_frame = tk.Frame(self.mode_selection_window)
        selection_frame.pack(side=tk.TOP, padx=10, pady=10)

        mode_selection_buttons = []
        for mode in SelectionMenuApp.Mode:
            button = tk.Button(selection_frame, text=mode.name, command=lambda v=mode.value: self.select_mode(v))
            button.pack(side=tk.BOTTOM)

            mode_selection_buttons.append(button)

    def close_mode_selection_window(self):
        self.mode_selection_window.withdraw()
        clear_widget(self.mode_selection_window)

    def select_mode(self, mode_val: int):
        self.selected_mode = SelectionMenuApp.Mode(mode_val)

        self.close_mode_selection_window()
        clear_window(self.window)
        self.close(window_keepalive=True)

        if self.selected_mode == SelectionMenuApp.Mode.Camera:
            self.instance = CameraApp(self.window, self.selected_camera)
            self.instance.update_stream()
        elif self.selected_mode == SelectionMenuApp.Mode.FormSetup:
            self.instance = FormSetupApp(self.window, self.selected_camera)
            self.instance.update_stream()
        elif self.selected_mode == SelectionMenuApp.Mode.CaptureResponses:
            self.instance = CaptureResponsesApp(self.window, self.selected_camera)
            self.instance.update_stream()
        else:
            raise ValueError("Invalid mode passed to select_mode!")

    def update_stream(self):
        frame_rets = []
        for device in self.capture_devices:
            ret, frame = device.read()
            frame_rets.append((ret, frame))

        ran_once = False
        for i, frame_ret in enumerate(frame_rets):
            ret, frame = frame_ret
            if ret:
                label = self.video_stream_labels[i]
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image)
                image = image.resize(SelectionMenuApp.PREVIEW_STREAM_DIM, Image.LANCZOS)

                photo = ImageTk.PhotoImage(image)
                label.config(image=photo)
                label.image = photo

                if not ran_once:
                    label.after(10, self.update_stream)
                    ran_once = True

    def close(self, window_keepalive=False):
        for device in self.capture_devices:
            if window_keepalive and device == self.selected_camera:
                continue

            device.release()

        if window_keepalive:
            return
        elif self.instance is not None:
            self.instance.close()

        try:
            self.window.destroy()
        except tk.TclError:
            pass


tk_window = tk.Tk()
# app = CameraApp(tk_window)
# app.update_stream()

# setup = FormSetupApp(tk_window, device=get_capture_device(1))
# setup.update_stream()

setup = CaptureResponsesApp(tk_window, device=get_capture_device(1))
setup.update_stream()

# setup = SelectionMenuApp(tk_window)
# setup.update_stream()

tk_window.mainloop()

# Make sure to close the camera stream when we exit
atexit.register(setup.close)

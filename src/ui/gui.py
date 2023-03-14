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
    device.set(3, device.get(cv2.CAP_PROP_FRAME_WIDTH))
    device.set(4, device.get(cv2.CAP_PROP_FRAME_HEIGHT))

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


class CameraApp:
    PROCESSING_DIM = (640, 480)
    ORIENTATION_TAG_ID = 0

    PROCESSING_CLIP_LIMIT = 2.0
    PROCESSING_TILE_GRID_SIZE = (8, 8)

    BUBBLE_DETECTION_BASELINE = 0.2

    class Position(enum.Enum):
        TL = enum.auto()
        TR = enum.auto()
        BL = enum.auto()
        BR = enum.auto()

    def __init__(self, window: tk.Tk):
        self.window = window
        self.window.title("TTScout")

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.snapshot_frame = None

        self.snapshot_button = tk.Button(self.window, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.cap = get_capture_device(0)
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

                print(i, len(approx_polygon))

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

    def locate_bubbles(self, grayscale: np.ndarray):
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(grayscale)

        # TODO: make like all of this better
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
                self.locate_bubbles(
                    cv2.resize(corrected, dsize=CameraApp.PROCESSING_DIM, interpolation=cv2.INTER_LANCZOS4)
                )

    def confirm_preview(self):
        self.preview_window.withdraw()
        # TODO save to csv

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

            self.bubble_frame.destroy()
            self.bubble_name.destroy()
            self.bubble_bounding_box.destroy()

            # special case where select_bubble_button isn't always a required widget to be displayed
            if self.select_bubble_button is not None:
                self.select_bubble_button.destroy()

        def update_bounding_box(self, bounding_box: (int, int, int, int)):
            if not self.is_tk:
                raise RuntimeError("IllegalState! Cannot invoke update_bounding_box when not is_tk")

            self.bubble_bounding_box["state"] = "normal"
            self.bubble_bounding_box.delete("1.0", "1.end")
            self.bubble_bounding_box.insert("1.0", str(bounding_box))
            self.bubble_bounding_box["state"] = "disabled"

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
            bubble_bounding_box["state"] = "disabled"
            bubble_bounding_box.pack(side=tk.TOP)

            return bubble_frame, bubble_name, bubble_bounding_box

        def tk_repr(self, container_frame: tk.Frame):
            bubble_frame, bubble_name, bubble_bounding_box = FormSetupApp.Bubble._make_tk_repr(
                container_frame,
                self.name,
                f"({self.rect_x}, {self.rect_y}, {self.rect_x + self.rect_w}, {self.rect_y + self.rect_h})"
            )

            # disable editing bubble name when making tk_repr instead of adding a new bubble
            bubble_name["state"] = "disabled"

            self._set_tk_props(bubble_frame, bubble_name, bubble_bounding_box)

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
                     delete_button_frame: tk.Frame = None,
                     delete_button: tk.Button = None):
            self.name = name
            self.bubbles = bubbles

            self.is_tk: bool = is_tk
            self.question_frame: tk.Frame | None = question_frame
            self.question_name: tk.Text | None = question_name
            self.bubbles_frame: tk.Frame | None = bubbles_frame
            self.delete_button_frame: tk.Frame | None = delete_button_frame
            self.delete_button: tk.Button | None = delete_button

        def __repr__(self):
            return f"Question(name={self.name}, bubbles={self.bubbles})"

        def _set_tk_props(self,
                          question_frame: tk.Frame,
                          question_name: tk.Text,
                          bubbles_frame: tk.Frame,
                          delete_button_frame: tk.Frame,
                          delete_button: tk.Button):
            self.is_tk = True

            self.question_frame = question_frame
            self.question_name = question_name
            self.bubbles_frame = bubbles_frame
            self.delete_button_frame = delete_button_frame
            self.delete_button = delete_button

        def delete_tk_widgets(self):
            if not self.is_tk:
                raise RuntimeError("IllegalState! Cannot invoke _delete_tk_widgets when not is_tk")

            for bubble in self.bubbles:
                bubble.delete_tk_widgets()

            self.question_frame.destroy()
            self.question_name.destroy()
            self.bubbles_frame.destroy()
            self.delete_button_frame.destroy()
            self.delete_button.destroy()

        def _delete_question(self, name: str, deletion_callback: Callable[[str], Any]):
            self.delete_tk_widgets()
            deletion_callback(name)

        @classmethod
        def _make_tk_repr(cls,
                          parent_frame: tk.Frame,
                          name: str,
                          bubbles: [FormSetupApp.Bubble]):
            question_frame = tk.Frame(parent_frame)
            question_frame.pack(side=tk.TOP)

            question_name = tk.Text(question_frame, width=40, height=1)
            question_name.insert("1.0", name)
            question_name["state"] = "disabled"
            question_name.pack(side=tk.TOP)

            bubbles_frame = tk.Frame(question_frame)
            bubbles_frame.pack(side=tk.RIGHT)

            delete_button_frame = tk.Frame(question_frame)
            delete_button_frame.pack(side=tk.LEFT)

            for bubble in bubbles:
                bubble.tk_repr(bubbles_frame)

            return question_frame, question_name, bubbles_frame, delete_button_frame

        def tk_repr(self, container_frame: tk.Frame, deletion_callback: str | Callable[[str], Any]):
            question_frame, question_name, bubbles_frame, delete_button_frame = self._make_tk_repr(
                container_frame, self.name, self.bubbles
            )

            delete_button = tk.Button(
                delete_button_frame, text="Delete", command=lambda: self._delete_question(self.name, deletion_callback)
            )
            delete_button.pack(side=tk.RIGHT)

            self._set_tk_props(question_frame, question_name, bubbles_frame, delete_button_frame, delete_button)

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

    def __init__(self, window: tk.Tk):
        super().__init__(window)
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

        self.shown_questions_container: tk.Frame | None = None
        self.shown_questions_scroll_frame: ScrollFrame | None = None
        self.curr_question_setting_bubbles = []

    def display_question(self, question: Question):
        if self.shown_questions_scroll_frame is not None:
            question.tk_repr(self.shown_questions_scroll_frame.viewPort, lambda n: self.remove_question(n))

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
        self.make_setup_window(Image.fromarray(self.setup_image), dims=CameraApp.PROCESSING_DIM)

    def close_setup_window(self):
        self.setup_window.withdraw()
        clear_widget(self.setup_window)
        self.setup_image = None

    def add_question_btn_callback(self):
        self.make_question_settings_window()

    def make_question_settings_window(self):
        self.question_settings_window.protocol("WM_DELETE_WINDOW", self.close_question_settings_window_callback)

        # Deiconify/Show the window
        if self.question_settings_window.winfo_exists():
            self.question_settings_window.deiconify()

        self.settings_frame = tk.Frame(self.question_settings_window)
        self.settings_frame.pack(side=tk.TOP, padx=10, pady=10)

        question_name = tk.Text(self.settings_frame, width=40, height=1)
        question_name.insert("1.0", "Enter Question Name Here...")
        question_name.pack(side=tk.TOP)

        add_bubble_button = tk.Button(self.settings_frame, text="Add Bubble", command=self.add_bubble_btn_callback)
        add_bubble_button.pack(side=tk.LEFT)

        confirm_button = tk.Button(
            self.question_settings_window,
            text="Confirm",
            command=lambda: self.close_question_settings_window_callback(
                question_name.get("1.0", "1.end"), self.curr_question_setting_bubbles
            )
        )
        confirm_button.pack(side=tk.RIGHT)

    def add_bubble_btn_callback(self):
        bubble = FormSetupApp.Bubble.from_tk(self.settings_frame, self.select_bubble_btn_callback)
        self.curr_question_setting_bubbles.append(bubble)

    def close_question_settings_window_callback(self, question_name: str = None, question_bubbles: [Bubble] = None):
        if question_name is not None and question_bubbles is not None:
            copied_bubbles = question_bubbles.copy()  # need to make a copy here, so it doesn't get gc 'ed early
            for bubble in copied_bubbles:
                bubble.update_name()

            question = FormSetupApp.Question(question_name, copied_bubbles)
            self.add_question_direct(question)

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
        preview_text["state"] = "disabled"
        preview_text.pack(side=tk.LEFT)

        _baseline_percent = 100 * CameraApp.BUBBLE_DETECTION_BASELINE

        def update_preview():
            while self.canvas_preview_update:
                cropped_img = crop(img, self.setup_canvas.rect_absolute)

                tk_img = ImageTk.PhotoImage(cropped_img)
                preview_label.config(image=tk_img)
                preview_label.image = tk_img

                rect_x0, rect_y0, rect_x1, rect_y1 = self.setup_canvas.rect_absolute
                extracted_img: np.ndarray = img_mat[rect_y0:rect_y1, rect_x0:rect_x1]

                n_px = (rect_x1 - rect_x0) * (rect_y1 - rect_y0)
                n_white_px = np.count_nonzero(extracted_img)
                prop_white = n_white_px / n_px

                preview_text["state"] = "normal"
                preview_text.delete("1.0", "1.end")
                preview_text.insert("1.0", f"{(100 * prop_white):3.3f}% white; baseline={_baseline_percent}")
                preview_text["state"] = "disabled"

                time.sleep(0.1)

        self.canvas_preview_update = True
        self.canvas_preview_update_thread = threading.Thread(target=update_preview)
        self.canvas_preview_update_thread.start()

        self.setup_canvas.create_image(0.5 * photo_width, 0.5 * photo_height, image=photo)
        self.setup_canvas.image = photo  # Keep reference to avoid garbage collection

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
        print(self.setup_canvas.rect_absolute)
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
    def __init__(self, window: tk.Tk):
        super().__init__(window)
        try:
            with open(FormSetupApp.JSON_PATH, "r") as input_json:
                self.data = FormSetupApp.Data.from_json(json.loads(input_json.read()))
        except FileNotFoundError:
            raise RuntimeError("Cannot start CaptureResponsesApp when data.json does not exist!")
        except EnvironmentError as environ_err:
            raise RuntimeError(environ_err)

    @classmethod
    def check_bubble_center_completeness(cls, bubble: np.ndarray):
        for row in bubble:
            if len(consecutive(row, step_size=0)) > 3:
                return False

        return True

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
        return n_labels <= 1

    def locate_bubbles(self, grayscale: np.ndarray):
        threshold = self.process_snapshot(grayscale)
        disp_copy = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2RGB)

        marked_bubbles = []
        for question in self.data.questions:
            for bubble in question.bubbles:
                rect_x0, rect_y0 = bubble.rect_x, bubble.rect_y
                rect_x1, rect_y1 = rect_x0 + bubble.rect_w, rect_y0 + bubble.rect_h
                extracted_img: np.ndarray = threshold[rect_y0:rect_y1, rect_x0:rect_x1]

                # cv2.rectangle(disp_copy, (rect_x0, rect_y0), (rect_x1, rect_y1), Color.GREEN.rgb)
                # cv2.circle(copy, (int(circle_x), int(circle_y)), int(radius), Color.RED.rgb)

                flat_nonzero = np.flatnonzero(extracted_img) % bubble.rect_w
                if len(flat_nonzero) > 0:
                    leftmost_y = np.min(flat_nonzero)
                    rightmost_y = np.max(flat_nonzero)

                    # print(f"ly: {leftmost_y}, ry: {rightmost_y}")

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

                if meets_all_conditions:
                    cv2.rectangle(disp_copy, (rect_x0, rect_y0), (rect_x1, rect_y1), Color.GREEN.rgb)
                    marked_bubbles.append(meets_all_conditions)
                else:
                    cv2.rectangle(disp_copy, (rect_x0, rect_y0), (rect_x1, rect_y1), Color.RED.rgb)

        self.img_window([Image.fromarray(disp_copy)], dims=CameraApp.PROCESSING_DIM)
        # self.img_window([Image.fromarray(shadow_filtered_image)], dims=CameraApp.PROCESSING_DIM)


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

        self.instance = None

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
            stream_label.pack()

            select_camera_button = tk.Button(self.window, text="Select", command=lambda: self.select_camera(i))
            select_camera_button.pack(side=tk.BOTTOM, padx=10, pady=10)

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
            self.instance = CameraApp(self.window)
            self.instance.update_stream()
        elif self.selected_mode == SelectionMenuApp.Mode.FormSetup:
            self.instance = FormSetupApp(self.window)
            self.instance.update_stream()
        elif self.selected_mode == SelectionMenuApp.Mode.CaptureResponses:
            self.instance = CaptureResponsesApp(self.window)
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
            device.release()

        if window_keepalive:
            return

        try:
            self.window.destroy()
        except tk.TclError:
            pass


tk_window = tk.Tk()
# setup = FormSetupApp(tk_window)
# setup.update_stream()

# app = CameraApp(tk_window)
# app.update_stream()
# setup = CaptureResponsesApp(tk_window)
# setup.update_stream()

setup = SelectionMenuApp(tk_window)
setup.update_stream()

tk_window.mainloop()

# Make sure to close the camera stream when we exit
atexit.register(setup.close)

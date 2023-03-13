from __future__ import annotations
from typing import Any, Callable, TextIO

import math
import json
import tkinter as tk
import warnings
import atexit
import enum

import cv2
import numpy as np
import pupil_apriltags
from PIL import Image, ImageTk
from pupil_apriltags import Detector

from util import four_point_transform, Color
from canvas import Canvas


def get_capture_device(source):
    device = cv2.VideoCapture(source)
    if device is None or not device.isOpened():
        warnings.warn(f"Unable to open VideoCapture Stream on Source: {source}")
        return None

    # set to maximum resolution
    device.set(3, device.get(cv2.CAP_PROP_FRAME_WIDTH))
    device.set(4, device.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return device


def clear_window(window: tk.BaseWidget, exclude: tk.BaseWidget | [tk.BaseWidget] = None):
    if window.winfo_exists():
        for widget in window.winfo_children():
            if (type(exclude) is list and widget not in exclude) or (widget is not exclude):
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


class CameraApp:
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

    def contour_apriltags(self, image: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        tag_pts = [tag.center.astype(int) for tag in tags if tag.tag_id in self.possible_tags]
        for tag in tags:
            if tag.tag_id in self.possible_tags or tag.tag_id == 0:
                pts = tag.corners.astype(int)
                cv2.polylines(image, [pts], True, Color.GREEN.rgb, 2)
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

            # Display the centers of each tag
            for tag in tags:
                cv2.circle(image, tag.center.astype(int), 5, Color.BLUE.rgb, -1)

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
            image = image.resize((640, 480), Image.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.video_stream.config(image=photo)
            self.video_stream.image = photo

        self.video_stream.after(10, self.update_stream)

    def clear_preview(self):
        clear_window(self.preview_window)

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.clear_preview()
            if self.snapshot_frame is not None:
                # self.show_preview()
                raw, grayscale = self.contour_edges(self.snapshot_frame)
                self.locate_bubbles(grayscale)

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
                          select_bubble_button: tk.Button):
            self.is_tk = True

            self.bubble_frame = bubble_frame
            self.bubble_name = bubble_name
            self.bubble_bounding_box = bubble_bounding_box
            self.select_bubble_button = select_bubble_button

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
        def from_tk(cls, settings_frame: tk.Frame, selection_callback: str | Callable[[FormSetupApp.Bubble], Any]):
            inst = cls("Enter Bubble Name Here...", 0, 0, 0, 0)

            bubble_frame = tk.Frame(settings_frame)
            bubble_frame.pack(side=tk.TOP)

            bubble_name = tk.Text(bubble_frame, width=40, height=1)
            bubble_name.insert("1.0", inst.name)
            bubble_name.pack(side=tk.TOP)

            bubble_bounding_box = tk.Text(bubble_frame, width=40, height=1)
            bubble_bounding_box.insert("1.0", "Missing Bounding Box!")
            bubble_bounding_box["state"] = "disabled"
            bubble_bounding_box.pack(side=tk.TOP)

            select_bubble_button = tk.Button(bubble_frame, text="Select", command=lambda: selection_callback(inst))
            select_bubble_button.pack(side=tk.RIGHT)

            inst._set_tk_props(bubble_frame, bubble_name, bubble_bounding_box, select_bubble_button)
            return inst

        @classmethod
        def from_json(cls, input_json: dict):
            return cls(**input_json)

    class Question(JsonSerializable):
        def __init__(self, name, bubbles: [FormSetupApp.Bubble]):
            self.name = name
            self.bubbles = bubbles

        def __repr__(self):
            return f"Question(name={self.name}, bubbles={self.bubbles})"

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

        self.setup_image = None
        self.settings_frame = None

        self.curr_question_setting_bubbles = []

    def add_question(self, question: str, bubbles: [Bubble]):
        existing_question = self.get_question(question)
        if existing_question is None:
            self.data.questions.append(FormSetupApp.Question(question, bubbles))

    def add_question_direct(self, question: Question):
        existing_question = self.get_question(question.name)
        if existing_question is None:
            self.data.questions.append(question)

    def get_question(self, question_name: str) -> [Bubble]:
        return next((question for question in self.data.questions if question_name == question.name), None)

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

                self.setup_image = raw_transform
                self.img_window([Image.fromarray(raw_transform)], dims=(640, 480))

    def confirm_preview(self):
        self.preview_window.withdraw()
        self.clear_preview()

        self.snapshot_frame = None
        # self.img_window([Image.fromarray(self.setup_frame)], dims=(640, 480))
        self.make_setup_window(Image.fromarray(self.setup_image), dims=(640, 480))

    def close_setup_window(self):
        self.setup_window.withdraw()
        clear_window(self.setup_window)
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
        clear_window(self.question_settings_window)

    def select_bubble_btn_callback(self, bubble: Bubble):
        self.make_canvas_window(Image.fromarray(self.setup_image), bubble, dims=(640, 480))

    def make_canvas_window(self, canvas_image: Image, bubble: Bubble, dims=(240, 320)):
        self.canvas_container_window.protocol("WM_DELETE_WINDOW", self.close_canvas_window_callback)

        # Deiconify/Show the window
        if self.canvas_container_window.winfo_exists():
            self.canvas_container_window.deiconify()

        # Create a PhotoImage object for the setup_image
        img = canvas_image.resize(dims, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        photo_width, photo_height = photo.width(), photo.height()

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
        clear_window(self.canvas_container_window, exclude=self.setup_canvas)

    def confirm_canvas_callback(self, bubble: Bubble):
        print(self.setup_canvas.rect_absolute)
        bubble.update_bounding_box(self.setup_canvas.rect_absolute)
        self.close_canvas_window_callback()

    def close(self):
        try:
            with open(FormSetupApp.JSON_PATH, "w", encoding="utf-8") as output_json:
                self.data.to_json_file(output_json)
        except EnvironmentError as environ_err:
            raise RuntimeError(environ_err)

        super().close()


tk_window = tk.Tk()
setup = FormSetupApp(tk_window)
setup.update_stream()

# app = CameraApp(tk_window)
# app.update_stream()
tk_window.mainloop()

# Make sure to close the camera stream when we exit
atexit.register(setup.close)

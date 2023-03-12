from __future__ import annotations

import tkinter as tk
import warnings
import atexit
import enum

import cv2
import numpy as np
import pupil_apriltags
from PIL import Image, ImageTk
from pupil_apriltags import Detector

from util import four_point_transform


def get_capture_device(source):
    device = cv2.VideoCapture(source)
    if device is None or not device.isOpened():
        warnings.warn(f"Unable to open VideoCapture Stream on Source: {source}")
        return None

    return device


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


# TODO is there a better way of doing this or is this already in an lib somewhere
# TODO also don't like how I have to use the rgb property, I wish it would just be
#  Color.RED and it would just pass in a tuple but you'd have to do either
#  Color.RED.value or in this case (somewhat better) Color.RED.rgb

class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    @property
    def rgb(self):
        return self.value


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

    def img_window(self, images, dims=(320, 240)):
        self.preview_window.protocol("WM_DELETE_WINDOW", self.disable_event)

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

    def define_edges(self, image):
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

            # Calculate the rotation matrix

            # TODO Karthik i got this working on hopes and dreams could u make it use a dictionary so that we dont
            #  need this stupid if statement thing.

            # TODO Done...maybe? Let me know if this is ok now

            tl_tag: pupil_apriltags.Detection = self.get_tag_at_pos(tags, self.Position.TL)
            tr_tag: pupil_apriltags.Detection = self.get_tag_at_pos(tags, self.Position.TR)

            if tl_tag is None or tr_tag is None:
                # If missing at least 1 tag, return early since we can't compute the rect
                # should never be missing more than 1 tag
                return

            tl_x, tl_y = tl_tag.center.astype(int)
            tr_x, tr_y = tr_tag.center.astype(int)

            # for tag in tags:
            #     if tag.tag_id == 1:
            #         x1, y1 = tag.center.astype(int)
            #     elif tag.tag_id == 6:
            #         x6, y6 = tag.center.astype(int)

            # TODO still some issues with rotated rects as I think you are only using TopLeft and TopRight tags
            #  maybe something for Karthik to do?

            rot = -np.rad2deg(np.arctan2((tr_y - tl_y), (tr_x - tl_x)))
            rot_mat = cv2.getRotationMatrix2D(center, rot, 1.0)

            # Transform the corners of the box
            corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            rotated_corners = cv2.transform(np.array([corners], dtype=np.float32), rot_mat)[0].astype(np.int32)

            # Draw the rotated rectangle
            cv2.drawContours(image, [rotated_corners], -1, Color.GREEN.rgb, 2)

    def contour_edges(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # find contours using edge biased image from canny, RETR_EXTERNAL for only outer hierarchical contours
        # CHAIN_APPROX_SIMPLE as we expect a paper to be constructed of lines, so we only need to store the 2 points
        # on the extremes of the line contour

        # see https://docs.opencv.org/4.7.0/d4/d73/tutorial_py_contours_begin.html
        # and https://docs.opencv.org/4.7.0/d9/d8b/tutorial_py_contours_hierarchy.html
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygon = None
        if len(contours) > 0:
            # sort from greatest to the least contour area (our paper should be the biggest contour in image)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in sorted_contours:
                # find perimeter of contour
                contour_perimeter = cv2.arcLength(contour, True)
                # simplify the contour down to polygonal curves, using epsilon as 0.02 * perimeter
                # see https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
                # and https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
                approx_polygon = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)

                # rects have 4 vertices so if this has 4 points/vertices then we assume we found paper
                if len(approx_polygon) == 4:
                    polygon = approx_polygon
                    break

            if polygon is None:
                return

            raw_transform = four_point_transform(image, polygon.reshape(4, 2))
            print([cv2.contourArea(c) for c in sorted_contours])

            self.img_window([Image.fromarray(edged),
                             Image.fromarray(cv2.drawContours(image, contours, -1, Color.GREEN.rgb, 3)),
                             Image.fromarray(raw_transform)])

    def update_stream(self):
        ret, frame = self.cap.read()

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect AprilTags in the frame
            self.define_edges(image)

            image = Image.fromarray(image)
            image = image.resize((640, 480), Image.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.video_stream.config(image=photo)
            self.video_stream.image = photo

        self.video_stream.after(10, self.update_stream)

    def clear_preview(self):
        if self.preview_window.winfo_exists():
            for widget in self.preview_window.winfo_children():
                widget.destroy()

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.clear_preview()
            if self.snapshot_frame is not None:
                # self.show_preview()
                self.contour_edges(self.snapshot_frame)

    def disable_event(self):
        pass

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


tk_window = tk.Tk()
app = CameraApp(tk_window)
app.update_stream()
tk_window.mainloop()

# Make sure to close the camera stream when we exit
atexit.register(app.close)

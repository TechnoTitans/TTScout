import warnings

import cv2
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk
from pupil_apriltags import Detector


def get_capture_device(source):
    device = cv2.VideoCapture(source)
    if device is None or not device.isOpened():
        warnings.warn(f"Unable to open VideoCapture Stream on Source: {source}")
        return None

    return device


class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("TTScout")

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.snapshot_button = tk.Button(self.window, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.cap = get_capture_device(0)
        if self.cap is None:
            exit(1)

        self.video_stream = tk.Label(self.video_frame)
        self.video_stream.pack()

        self.preview_window = tk.Toplevel(self.window)
        self.preview_window.withdraw()

        self.detector = Detector(
            families='tag16h5',
            nthreads=1,
            quad_decimate=1,
            quad_sigma=1,
            refine_edges=5,
            decode_sharpening=0.25,
            debug=0
        )

    def define_edges(self, image):
        possible_tags = (1, 6, 8, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        # Draw AprilTag outlines on the frame
        tag_pts = []
        for tag in tags:
            if tag.tag_id not in possible_tags and tag.tag_id != 0:
                break
            pts = tag.corners.astype(int)
            if tag.tag_id in possible_tags:
                tag_pts.append(tag.center.astype(int))
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        if len(tag_pts) >= 4:
            # Find the top-left and bottom-right corners of the box
            x_vals = [pt[0] for pt in tag_pts]
            y_vals = [pt[1] for pt in tag_pts]
            x_min = min(x_vals)
            y_min = min(y_vals)
            x_max = max(x_vals)
            y_max = max(y_vals)

            # Draw the box with the centers of the tags
            # center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)

            x1 = 0
            y1 = 0
            x6 = 0
            y6 = 0

            for tagPts in tags:
                if tagPts.tag_id == 1:
                    x1 = tagPts.center[0].astype(int)
                    y1 = tagPts.center[1].astype(int)
                elif tagPts.tag_id == 6:
                    x6 = tagPts.center[0].astype(int)
                    y6 = tagPts.center[1].astype(int)

            rot = -np.rad2deg(np.arctan2((y6 - y1), (x6 - x1)))
            rot_mat = cv2.getRotationMatrix2D(center, rot, 1.0)
            corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            corners = corners.reshape((-1, 1, 2))
            rotated_corners = cv2.transform(corners, rot_mat)
            rotated_corners = rotated_corners.reshape((4, 2))
            rotated_corners = np.intp(rotated_corners)

            # Draw the rotated rectangle
            cv2.polylines(image, [rotated_corners], True, (0, 255, 0), 2)

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
        for widget in self.preview_window.winfo_children():
            widget.destroy()

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.clear_preview()
            self.show_preview()

    def show_preview(self):
        image = Image.fromarray(self.snapshot_frame)
        image = image.resize((320, 240), Image.LANCZOS)

        tags = self.detector.detect(image)

        # Draw AprilTag outlines on the frame
        for tag in tags:
            pts = tag.corners.astype(int)
            print(tag.tag_id)
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        photo = ImageTk.PhotoImage(image)

        self.preview_window.deiconify()
        preview_label = tk.Label(self.preview_window, image=photo)
        preview_label.pack()

        confirm_button = tk.Button(self.preview_window, text="Confirm", command=self.confirm_preview)
        confirm_button.pack(side=tk.LEFT)

        retry_button = tk.Button(self.preview_window, text="Retry", command=self.retry_preview)
        retry_button.pack(side=tk.RIGHT)

    def confirm_preview(self):
        self.preview_window.withdraw()

    def retry_preview(self):
        self.preview_window.withdraw()
        self.snapshot_frame = None

    def close(self):
        self.cap.release()
        self.window.destroy()


window = tk.Tk()
app = CameraApp(window)
app.update_stream()
window.mainloop()

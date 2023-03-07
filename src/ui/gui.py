import cv2
import warnings
import tkinter as tk
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
        self.window.title("Camera App")

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.snapshot_frame = None

        self.snapshot_button = tk.Button(self.window, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.cap = get_capture_device(12)
        if self.cap is None:
            exit(1)

        self.video_stream = tk.Label(self.video_frame)
        self.video_stream.pack()

        self.preview_window = tk.Toplevel(self.window)
        self.preview_window.withdraw()

        self.detector = Detector(
            families='tag16h5',
            nthreads=1,
            quad_decimate=10,
            quad_sigma=1,
            refine_edges=1,
            decode_sharpening=0.35,
            debug=0
        )

    def update_stream(self):
        ret, frame = self.cap.read()

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect AprilTags in the frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)

            # Draw AprilTag outlines on the frame
            for tag in tags:
                pts = tag.corners.astype(int)
                print(tag.tag_id)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)

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

        photo = ImageTk.PhotoImage(image)

        self.preview_window.deiconify()
        preview_label = tk.Label(self.preview_window, image=photo)
        preview_label.image = photo
        preview_label.pack()

        confirm_button = tk.Button(self.preview_window, text="Confirm", command=self.close_preview)
        confirm_button.pack(side=tk.LEFT)

        retry_button = tk.Button(self.preview_window, text="Retry", command=self.retry_preview)
        retry_button.pack(side=tk.RIGHT)

    def close_preview(self):
        self.preview_window.withdraw()

    def retry_preview(self):
        self.preview_window.withdraw()
        self.snapshot_frame = None

    def close(self):
        self.cap.release()
        self.window.destroy()


tkWindow = tk.Tk()
app = CameraApp(tkWindow)
app.update_stream()
tkWindow.mainloop()

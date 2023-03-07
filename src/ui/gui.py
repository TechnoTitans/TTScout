import cv2
import tkinter as tk
from PIL import Image, ImageTk


class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Camera App")

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.snapshot_button = tk.Button(self.window, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)

        self.video_stream = tk.Label(self.video_frame)
        self.video_stream.pack()

        self.preview_window = tk.Toplevel(self.window)
        self.preview_window.withdraw()

        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack()

    def update_stream(self):
        ret, frame = self.cap.read()

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            image = image.resize((640, 480), Image.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.video_stream.config(image=photo)
            self.video_stream.image = photo

        self.video_stream.after(10, self.update_stream)

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            self.snapshot_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_preview()

    def show_preview(self):
        image = Image.fromarray(self.snapshot_frame)
        image = image.resize((320, 240), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image)

        self.preview_label.config(image="")
        self.preview_label.config(image=photo)
        self.preview_label.image = photo

        self.preview_window.deiconify()
        confirm_button = tk.Button(self.preview_window, text="Confirm", command=self.close_preview)
        confirm_button.pack(side=tk.LEFT)

        retry_button = tk.Button(self.preview_window, text="Retry", command=self.retry_preview)
        retry_button.pack(side=tk.RIGHT)

        self.update_stream()

    def close_preview(self):
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

from cv2 import cv2
import tkinter as tk
from PIL import Image, ImageTk


class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("TTScout")
        self.window.resizable(width=False, height=False)

        self.snapshot_frame = None

        self.video_frame = tk.Frame(self.window)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.snapshot_button = tk.Button(self.window, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)
        self.video_stream = tk.Label(self.video_frame)
        self.video_stream.pack()

        self.preview_window = tk.Toplevel(self.window)
        self.preview_window.resizable(width=False, height=False)
        self.preview_window.withdraw()

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
        self.preview_window.title("Preview")

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
        self.snapshot_frame = None

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

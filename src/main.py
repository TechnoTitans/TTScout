import tkinter as tk
import atexit
import src.ui.gui as gui


def try_cast_as_int(string: str):
    try:
        return int(string)
    except ValueError as err:
        print(f"ValueError trying to parse string as int: {err}")
        return None


if __name__ == "__main__":
    selected_mode = input(f"Select a mode ({', '.join(gui.MODES)}): ")

    lowercase_selected_mode = selected_mode.lower()
    tk_window = tk.Tk()

    if selected_mode == "CameraApp":
        app = gui.launchCameraApp(tk_window)
    elif selected_mode == "FormSetupApp":
        app = gui.launchFormSetupApp(tk_window)
    elif selected_mode == "CaptureResponsesApp":
        input_device_id_str = input("Select capture device [0, n]: ")
        input_device_id = try_cast_as_int(input_device_id_str)

        if input_device_id is None:
            raise RuntimeError(f"Failed to cast string \"{input_device_id_str}\" to int!")

        input_device = gui.get_capture_device(input_device_id)
        if input_device is None:
            raise RuntimeError(f"Failed to load capture device: {input_device_id}")

        app = gui.launchCaptureResponsesApp(tk_window, input_device)
    elif selected_mode == "SelectionMenuApp" or len(selected_mode) == 0:
        # special case to just fast-forward to the selection menu
        app = gui.launchSelectionMenuApp(tk_window)
    else:
        raise RuntimeError("Invalid option!")

    atexit.register(app.close)
    tk_window.mainloop()

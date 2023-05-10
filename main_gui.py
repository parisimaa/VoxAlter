import tkinter as tk
import os

class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Main Window")
        self.create_buttons()
        self.current_window = None

    def create_buttons(self):
        def run_pitch_shift():
            os.system('python pitch_shift_mic.py')

        whisper_button = tk.Button(self.master, text="Whisper", command=self.open_whisper_window)
        whisper_button.pack(pady=10)
        
        robot_button = tk.Button(self.master, text="Robot", command=self.open_robot_window)
        robot_button.pack(pady=10)
        
        phase_shift_button = tk.Button(self.master, text="Phase shift", command=run_pitch_shift)
        phase_shift_button.pack(pady=10)
        
        quit_button = tk.Button(self.master, text="Quit", command=self.master.quit)
        quit_button.pack(pady=10)

    def open_whisper_window(self):
        if self.current_window is not None:
            return

        new_window = tk.Toplevel(self.master)
        new_window.title("Whisper")
        tk.Label(new_window, text="Whisper").pack(pady=10)
        quit_button = tk.Button(new_window, text="Quit", command=self.close_window(new_window))
        quit_button.pack(pady=10)
        self.current_window = new_window

    def open_robot_window(self):
        if self.current_window is not None:
            return

        new_window = tk.Toplevel(self.master)
        new_window.title("Robot")
        tk.Label(new_window, text="Robot").pack(pady=10)
        quit_button = tk.Button(new_window, text="Quit", command=lambda: self.close_window(new_window))
        quit_button.pack(pady=10)
        self.current_window = new_window

    def open_phase_shift_window(self):
        if self.current_window is not None:
            return

        new_window = tk.Toplevel(self.master)
        new_window.title("Phase shift")
        tk.Label(new_window, text="Phase shift").pack(pady=10)
        quit_button = tk.Button(new_window, text="Quit", command=lambda: self.close_window(new_window))
        quit_button.pack(pady=10)
        self.current_window = new_window

    def close_window(self, window):
        window.destroy()
        self.current_window = None

root = tk.Tk()
main_window = MainWindow(root)
root.mainloop()

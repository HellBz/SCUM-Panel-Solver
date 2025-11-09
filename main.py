# main.py
# English comments only.

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import pytesseract
import numpy as np
import json
from itertools import product
from capture import capture_scum_window


class PanelSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SCUM Panel Solver Capture")
        self.root.geometry("1280x800")
        self.root.configure(bg="#222")

        self.img_bgr = None
        self.result_overlay = None
        self.calibration = None
        self.zoom = 1.0
        self.entries = {}

        self.setup_ui()

        # Load calibration.json automatically
        try:
            with open("calibration.json", "r") as f:
                self.calibration = json.load(f)
                print("✅ Loaded calibration.json")
        except Exception as e:
            print("⚠️ No calibration.json found:", e)

    # ---------- UI ----------
    def setup_ui(self):
        left = tk.Frame(self.root, bg="#222")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Button(left, text="📷 Capture SCUM", command=self.capture_image, width=20).pack(pady=4)
        tk.Button(left, text="🖼 Load Image", command=self.load_image, width=20).pack(pady=4)
        tk.Button(left, text="🔍 Analyze", command=self.analyze, width=20).pack(pady=4)
        tk.Button(left, text="🧩 Solve Panel", command=self.solve_panel, width=20).pack(pady=8)

        self.inp_entry = self._make_field(left, "Input", "#1e4228")
        self.outa_entry = self._make_field(left, "Output A", "#422828")
        self.outb_entry = self._make_field(left, "Output B", "#283742")

        tk.Label(left, text="Slots (L / R)", fg="white", bg="#222",
                 font=("Arial", 11, "bold")).pack(pady=(10, 4))

        self.slot_frame = tk.Frame(left, bg="#222")
        self.slot_frame.pack(pady=4)

        # --- Replace entries with interactive slot buttons ---
        for i in range(1, 9):
            row = tk.Frame(self.slot_frame, bg="#222")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"S{i}", fg="white", bg="#222", width=3).pack(side="left")

            def make_slot(side, color):
                btn = tk.Button(row, text="+0", bg=color, fg="white", width=8, relief="raised")
                btn.pack(side="left", padx=2)
                btn.bind("<Button-1>", lambda e, key=f"{side}{i}": self.open_slot_menu(key, e))
                self.entries[f"{side}{i}"] = btn

            make_slot("L", "#330000")
            make_slot("R", "#001133")

        # ----- Scrollable Canvas -----
        canvas_frame = tk.Frame(self.root, bg="#111")
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="black")
        self.hbar = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.vbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def open_slot_menu(self, key, event):
        """Open a popup menu for selecting operator and value."""
        menu = tk.Toplevel(self.root)
        menu.overrideredirect(True)
        menu.configure(bg="#333")
        menu.geometry(f"+{event.x_root}+{event.y_root}")

        ops = ["+", "-", "*", "/"]
        vals = [ 0, 2, 10, 20, 40, 60, 80, 100]
        chosen = {"op": "+", "val": 1}

        def set_operator(op):
            chosen["op"] = op

        def set_value(val):
            chosen["val"] = val
            self.entries[key].config(text=f"{chosen['op']}{val}")
            menu.destroy()

        op_frame = tk.Frame(menu, bg="#333")
        op_frame.pack(pady=4)
        for op in ops:
            b = tk.Button(op_frame, text=op, width=3, command=lambda o=op: set_operator(o))
            b.pack(side="left", padx=2)

        val_frame = tk.Frame(menu, bg="#333")
        val_frame.pack(pady=4)
        for v in vals:
            b = tk.Button(val_frame, text=str(v), width=3,
                          command=lambda val=v: set_value(val))
            b.pack(side="left", padx=2, pady=2)

        # Close if clicked outside
        def close_menu(e):
            if not menu.winfo_containing(e.x_root, e.y_root):
                menu.destroy()
                self.root.unbind("<Button-1>", close_id)

        close_id = self.root.bind("<Button-1>", close_menu)

    def _make_field(self, parent, label, color):
        frame = tk.Frame(parent, bg="#222")
        frame.pack(fill="x", pady=3)
        tk.Label(frame, text=label, fg="white", bg="#222", width=9, anchor="w").pack(side="left")
        entry = tk.Entry(frame, bg=color, fg="white", justify="center")
        entry.pack(side="left", fill="x", expand=True)
        return entry

    # ---------- Canvas & Zoom ----------
    def _on_canvas_resize(self, event):
        if self.img_bgr is not None:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def display_image(self, img):
        if img is None:
            return
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w * self.zoom), int(h * self.zoom)))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.canvas.imgtk = imgtk
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.configure(scrollregion=(0, 0, resized.shape[1], resized.shape[0]))

    def on_zoom(self, event):
        if self.img_bgr is None:
            return
        self.zoom *= 1.1 if event.delta > 0 else 0.9
        self.zoom = max(0.3, min(3.0, self.zoom))
        current = self.result_overlay if self.result_overlay is not None else self.img_bgr
        self.display_image(current)

    # ---------- Image handling ----------
    def capture_image(self):
        import os, time
        img = capture_scum_window()
        if img is None:
            messagebox.showerror("Error", "SCUM window not found.")
            return

        os.makedirs("capture", exist_ok=True)
        filename = f"capture/capture_{int(time.time())}.png"
        cv2.imwrite(filename, img)
        print(f"💾 Saved capture to {filename}")

        self.img_bgr = img
        self.display_image(img)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        self.img_bgr = cv2.imread(path)
        self.display_image(self.img_bgr)

    # ---------- Unified OCR ----------
    def read_op_value_from_roi(self, img, roi, allow_auto_mul=True, op_mode=False):
        """Read combined operator+number like '+100', '/2', '*3', '-5', or plain numbers."""
        x1, y1, x2, y2 = roi
        crop = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        gray = cv2.convertScaleAbs(gray, alpha=2.8, beta=25)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 4)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)

        # choose whitelist depending on type
        if op_mode:
            # operator slot: only +, -, *, / and allowed digits
            config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=+-*/012468'
        else:
            # numeric field: allow full number range including negative and decimal
            config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.-'

        text = pytesseract.image_to_string(gray, config=config).strip()

        text = text.replace("O", "0").replace("o", "0")
        text = text.replace("I", "1").replace("l", "1")
        text = text.replace("×", "*").replace("x", "*").replace("X", "*")
        text = text.replace("÷", "/").replace("\\", "/")
        text = text.replace(" ", "")
        
        # Fix common OCR issues for '/' operators
        if op_mode and text.startswith(("1", "I", "l")) and len(text) > 1 and text[1:].isdigit():
            text = "/" + text[1:]

        # only auto-prepend * if allowed
        if allow_auto_mul and not op_mode and text and text[0].isdigit():
            text = "*" + text

        # if operator mode, enforce validity
        if op_mode:
            valid_ops = [f"{o}{v}" for o in ["+", "-", "*", "/"] for v in [2, 10, 20, 40, 60, 80, 100]]
            if text not in valid_ops:
                text = ""

        return text, crop


    # ---------- Analyze ----------
    def analyze(self):
        if self.img_bgr is None or self.calibration is None:
            messagebox.showerror("Error", "Load image and calibration first.")
            return

        img = self.img_bgr.copy()
        calib = self.calibration

        inp, _ = self.read_op_value_from_roi(img, calib["input"], allow_auto_mul=False, op_mode=False)
        outa, _ = self.read_op_value_from_roi(img, calib["output_a"], allow_auto_mul=False, op_mode=False)
        outb, _ = self.read_op_value_from_roi(img, calib["output_b"], allow_auto_mul=False, op_mode=False)


        self.inp_entry.delete(0, tk.END)
        self.inp_entry.insert(0, inp)
        self.outa_entry.delete(0, tk.END)
        self.outa_entry.insert(0, outa)
        self.outb_entry.delete(0, tk.END)
        self.outb_entry.insert(0, outb)

        for i, slot in enumerate(calib["slots"], start=1):
            l_op, _ = self.read_op_value_from_roi(img, slot["left_op"], op_mode=True)
            r_op, _ = self.read_op_value_from_roi(img, slot["right_op"], op_mode=True)
            self.entries[f"L{i}"].config(text=l_op or "+0")
            self.entries[f"R{i}"].config(text=r_op or "+0")

        self.result_overlay = img
        self.display_image(img)

    # ---------- Math helpers ----------
    def clean_op(self, op: str) -> str:
        allowed = "0123456789+-*/"
        return "".join(c for c in op if c in allowed)

    def apply_op(self, value: float, op: str) -> float:
        """Apply mathematical operator like '+10', '*2', '/3', '-5'."""
        if not op:
            return value
        try:
            if op.startswith('+'):
                return value + float(op[1:])
            elif op.startswith('-'):
                return value - float(op[1:])
            elif op.startswith('*'):
                return value * float(op[1:])
            elif op.startswith('/'):
                divisor = float(op[1:])
                if divisor != 0:
                    return value / divisor
        except Exception:
            pass
        return value

    def _try_float(self, val):
        try:
            return float(val)
        except ValueError:
            return 0.0

    # ---------- Solver ----------
    def solve_panel(self):
        if self.img_bgr is None or self.calibration is None:
            messagebox.showerror("Error", "No image/calibration.")
            return

        inp_val = self._try_float(self.inp_entry.get())
        outa_val = self._try_float(self.outa_entry.get())
        outb_val = self._try_float(self.outb_entry.get())

        slots = []
        for i in range(1, 9):
            l = self.clean_op(self.entries[f"L{i}"].cget("text").strip())
            r = self.clean_op(self.entries[f"R{i}"].cget("text").strip())
            slots.append((l, r))

        best_combo = None
        found_result = None

        for combo in product([0, 1], repeat=len(slots)):
            try:
                val_a = inp_val
                val_b = inp_val
                for active, (l_op, r_op) in zip(combo, slots):
                    if active:
                        if l_op:
                            val_a = self.apply_op(val_a, l_op)
                        if r_op:
                            val_b = self.apply_op(val_b, r_op)
                if abs(val_a - outa_val) < 0.5 and abs(val_b - outb_val) < 0.5:
                    best_combo = combo
                    found_result = (val_a, val_b)
                    break
            except Exception:
                continue

        img = self.img_bgr.copy()
        for i, slot in enumerate(self.calibration["slots"], start=1):
            cx, cy = slot["center"]
            color = (0, 255, 0) if best_combo and best_combo[i - 1] == 1 else (0, 0, 255)
            cv2.circle(img, (cx, cy), 18, color, -1)
            cv2.putText(img, f"S{i}", (cx - 20, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.result_overlay = img
        self.display_image(img)

        # --- NEW: colored dot after "S1..S8" labels ---
        for i in range(1, 9):
            # Remove old indicator if it exists
            if hasattr(self, f"slot_indicator_{i}"):
                getattr(self, f"slot_indicator_{i}").destroy()

            if best_combo:
                color = "#00ff00" if best_combo[i - 1] == 1 else "#ff0000"
                # Find the label widget (the one that says S1, S2, ...)
                for widget in self.slot_frame.winfo_children():
                    if isinstance(widget, tk.Frame):
                        children = widget.winfo_children()
                        if children and children[0].cget("text") == f"S{i}":
                            label = children[0]
                            dot = tk.Label(widget, text="●", fg=color, bg="#222", font=("Arial", 16, "bold"))
                            dot.pack(side="left", padx=(2, 6))  # directly after the text
                            setattr(self, f"slot_indicator_{i}", dot)
                            break

        if best_combo:
            print(f"✅ Combination found: {''.join(str(x) for x in best_combo)}")
            print(f"A={found_result[0]} (target {outa_val}), B={found_result[1]} (target {outb_val})")
        else:
            print("❌ No valid combination found.")
            # Remove indicators if no valid combination
            for i in range(1, 9):
                if hasattr(self, f"slot_indicator_{i}"):
                    getattr(self, f"slot_indicator_{i}").destroy()



if __name__ == "__main__":
    root = tk.Tk()
    app = PanelSolverApp(root)
    root.mainloop()

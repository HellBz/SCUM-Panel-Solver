# calibrate_slots.py
# English comments only.

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json


def find_ref_positions(image, upper_ref, lower_ref):
    """Find upper and lower reference screws using robust multi-scale template matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ref1 = cv2.cvtColor(upper_ref, cv2.COLOR_BGR2GRAY)
    ref1 = cv2.equalizeHist(ref1)
    ref2 = cv2.cvtColor(lower_ref, cv2.COLOR_BGR2GRAY)
    ref2 = cv2.equalizeHist(ref2)

    def find_best(ref, label):
        best_val, best_loc, best_scale = -1, (0, 0), 1.0
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            resized = cv2.resize(ref, (0, 0), fx=scale, fy=scale)
            res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val, best_loc, best_scale = max_val, max_loc, scale

        print(f"{label} match quality: {best_val:.3f} (scale {best_scale})")
        if best_val < 0.6:
            print(f"⚠️  {label} reference match is weak, check lighting or zoom.")
        if best_loc is None:
            raise RuntimeError(f"{label} reference could not be found.")

        h, w = ref.shape
        h, w = int(h * best_scale), int(w * best_scale)
        center = (best_loc[0] + w // 2, best_loc[1] + h // 2)
        return center, best_val

    center1, val1 = find_best(ref1, "Upper")
    center2, val2 = find_best(ref2, "Lower")

    # Save debug overlay instead of showing it
    dbg = image.copy()
    cv2.circle(dbg, center1, 20, (0, 255, 0), 3)
    cv2.circle(dbg, center2, 20, (0, 0, 255), 3)
    cv2.putText(dbg, f"{val1:.2f}", (center1[0] + 25, center1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(dbg, f"{val2:.2f}", (center2[0] + 25, center2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = os.path.join(os.path.dirname(__file__), "ref_debug.png")
    cv2.imwrite(out_path, dbg)
    print(f"🖼️ Saved reference debug image: {out_path}")

    return center1, center2


class Calibrator(tk.Tk):
    def __init__(self, img_path, upper_ref_path, lower_ref_path):
        super().__init__()
        self.title("SCUM Calibration Tool")
        self.state("zoomed")
        self.configure(bg="#111")

        base = os.path.dirname(os.path.abspath(__file__))
        self.img_bgr = cv2.imread(img_path)
        self.upper_ref = cv2.imread(os.path.join(base, upper_ref_path))
        self.lower_ref = cv2.imread(os.path.join(base, lower_ref_path))

        if self.img_bgr is None:
            raise FileNotFoundError("Image could not be loaded.")

        # Find reference positions
        self.upper, self.lower = find_ref_positions(self.img_bgr, self.upper_ref, self.lower_ref)
        self.slots = self._gen_slots(self.upper, self.lower)

        self.input_roi = None
        self.out_a = None
        self.out_b = None
        self.op_left = {}
        self.op_right = {}
        self.drag_index = None
        self.drag_offset = (0, 0)
        self.mode = None
        self.drag_rect = None
        self.temp_rect_start = None
        self.show_help = True  # show help overlay initially

        self.load_calibration()

        # UI setup
        self.canvas = tk.Canvas(self, bg="#222", highlightthickness=0)
        self.hbar = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.hbar.pack(side="bottom", fill="x")
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Key>", self._on_key)
        self.canvas.bind("<Button-3>", self._save_json)

        self._update_display()

    def load_calibration(self):
        """Load previous calibration.json if present."""
        base = os.path.dirname(os.path.abspath(__file__))
        cal_path = os.path.join(base, "calibration.json")
        if not os.path.exists(cal_path):
            print("ℹ️ No existing calibration found.")
            return

        with open(cal_path, "r") as f:
            data = json.load(f)

        self.upper = tuple(data["refs"]["upper_ref"])
        self.lower = tuple(data["refs"]["lower_ref"])
        self.input_roi = data.get("input")
        self.out_a = data.get("output_a")
        self.out_b = data.get("output_b")

        self.slots = []
        self.op_left = {}
        self.op_right = {}
        for i, slot in enumerate(data["slots"]):
            cx, cy = slot["center"]
            self.slots.append([cx, cy])
            if "left_op" in slot:
                self.op_left[i] = slot["left_op"]
            if "right_op" in slot:
                self.op_right[i] = slot["right_op"]

        print(f"✅ Loaded calibration with {len(self.slots)} slots.")

    def _gen_slots(self, upper, lower):
        dx = lower[0] - upper[0]
        dy = lower[1] - upper[1]
        step = dy / 7
        slots = []
        for i in range(8):
            cy = int(upper[1] + i * step)
            cx = int(upper[0] + dx * (i / 7))
            slots.append([cx, cy])
        return slots

    def _on_key(self, event):
        key = event.char.lower()
        if key == "i":
            self.mode = "input"
            print("🟩 Define Input ROI")
        elif key == "a":
            self.mode = "out_a"
            print("🟥 Define Output A ROI (Red)")
        elif key == "b":
            self.mode = "out_b"
            print("🟦 Define Output B ROI (Blue)")
        elif key == "l":
            self.mode = "op_left"
            print("🔵 Define Left Operator")
        elif key == "r":
            self.mode = "op_right"
            print("🔴 Define Right Operator")
        elif key == "h":
            self.show_help = not self.show_help
            print("ℹ️ Toggle help overlay:", self.show_help)
            self._update_display()
        else:
            self.mode = None

    def _on_click(self, event):
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        if self.mode in ["input", "out_a", "out_b", "op_left", "op_right"]:
            self.temp_rect_start = (x, y)
            return

        # dragging slots
        min_dist = 9999
        for i, (cx, cy) in enumerate(self.slots):
            dist = ((x - cx)**2 + (y - cy)**2)**0.5
            if dist < min_dist and dist < 50:
                self.drag_index = i
                self.drag_offset = (x - cx, y - cy)
                min_dist = dist

    def _on_drag(self, event):
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        if self.temp_rect_start and self.mode:
            x0, y0 = self.temp_rect_start
            self.drag_rect = (x0, y0, x, y)
            self._update_display()
            return

        if self.drag_index is not None:
            dx, dy = self.drag_offset
            self.slots[self.drag_index] = [x - dx, y - dy]
            self._update_display()

    def _on_release(self, event):
        if self.temp_rect_start and self.mode:
            x0, y0 = self.temp_rect_start
            x1 = int(self.canvas.canvasx(event.x))
            y1 = int(self.canvas.canvasy(event.y))
            roi = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

            if self.mode == "input":
                self.input_roi = roi
            elif self.mode == "out_a":
                self.out_a = roi
            elif self.mode == "out_b":
                self.out_b = roi
            elif self.mode == "op_left":
                idx = self._nearest_slot(y1)
                self.op_left[idx] = roi
            elif self.mode == "op_right":
                idx = self._nearest_slot(y1)
                self.op_right[idx] = roi

            print(f"📏 Set {self.mode} ROI: {roi}")
            self.temp_rect_start = None
            self.drag_rect = None
            self.mode = None
            self._update_display()
        self.drag_index = None

    def _nearest_slot(self, y):
        idx = 0
        min_d = 9999
        for i, (_, cy) in enumerate(self.slots):
            d = abs(cy - y)
            if d < min_d:
                min_d = d
                idx = i
        return idx

    def _draw_help_overlay(self, img):
        """Draw semi-transparent help overlay with key bindings and tips."""
        overlay = img.copy()
        x0, y0 = 20, 20
        x1, y1 = 520, 360
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        lines = [
            "Calibration Help (press H to hide/show)",
            "",
            "Open image: file dialog appears on start.",
            "Reference screws detected automatically.",
            "",
            "Keys:",
            "I → Input ROI (green box)",
            "A → Output A ROI (red box)",
            "B → Output B ROI (blue box)",
            "L → Left operator (blue beside slots)",
            "R → Right operator (red beside slots)",
            "",
            "Mouse:",
            "Left-drag → draw ROI / move slot centers",
            "Right-click → save calibration + overlay",
            "",
            "Files created:",
            "calibration.json (used by main.py)",
            "calibration_overlay.png (visual check)",
        ]

        y = y0 + 30
        for i, t in enumerate(lines):
            scale = 0.65 if i != 0 else 0.75
            thick = 1 if i != 0 else 2
            cv2.putText(img, t, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thick, cv2.LINE_AA)
            y += 24 if i != 0 else 28

    def _update_display(self):
        img = self._draw_debug()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, pil_img.width, pil_img.height))

    def _draw_debug(self):
        img = self.img_bgr.copy()
        cv2.circle(img, self.upper, 20, (0,165,255), 3)
        cv2.circle(img, self.lower, 20, (0,165,255), 3)
        cv2.putText(img, "Upper", (self.upper[0]+30, self.upper[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        cv2.putText(img, "Lower", (self.lower[0]+30, self.lower[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        for i, (cx, cy) in enumerate(self.slots):
            cv2.rectangle(img, (cx-40, cy-15), (cx+40, cy+15), (0,255,255), 1)
            cv2.putText(img, f"S{i+1}", (cx-20, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        for idx, roi in self.op_left.items():
            x0,y0,x1,y1 = roi
            cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 2)
            cv2.putText(img, f"L{idx+1}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        for idx, roi in self.op_right.items():
            x0,y0,x1,y1 = roi
            cv2.rectangle(img, (x0,y0), (x1,y1), (0,0,255), 2)
            cv2.putText(img, f"R{idx+1}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        for name, roi, color in [
            ("Input", self.input_roi, (0,255,0)),
            ("OutA", self.out_a, (0,0,255)),
            ("OutB", self.out_b, (255,0,0))
        ]:
            if roi:
                x0,y0,x1,y1 = roi
                cv2.rectangle(img, (x0,y0), (x1,y1), color, 2)
                cv2.putText(img, name, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.show_help:
            self._draw_help_overlay(img)

        if self.drag_rect:
            x0,y0,x1,y1 = self.drag_rect
            cv2.rectangle(img, (x0,y0), (x1,y1), (200,200,200), 1)
        return img

    def _save_json(self, event=None):
        data = {
            "refs": {"upper_ref": self.upper, "lower_ref": self.lower},
            "input": self.input_roi,
            "output_a": self.out_a,
            "output_b": self.out_b,
            "slots": []
        }
        for i, (cx, cy) in enumerate(self.slots):
            s = {"center": [cx, cy]}
            if i in self.op_left:
                s["left_op"] = self.op_left[i]
            if i in self.op_right:
                s["right_op"] = self.op_right[i]
            data["slots"].append(s)

        base = os.path.dirname(os.path.abspath(__file__))
        cal_path = os.path.join(base, "calibration.json")
        with open(cal_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved calibration: {cal_path}")

        overlay = self._draw_debug()
        cv2.imwrite(os.path.join(base, "calibration_overlay.png"), overlay)
        print("🖼️ Saved calibration_overlay.png for verification.")


if __name__ == "__main__":
    img_path = filedialog.askopenfilename(title="Select screenshot", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
    if not img_path:
        raise SystemExit("No image selected.")

    app = Calibrator(
        img_path=img_path,
        upper_ref_path="assets/upper_screw.png",
        lower_ref_path="assets/lower_screw.png"
    )
    app.mainloop()

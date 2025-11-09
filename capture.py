
# capture.py
# English comments only.
import win32gui
import win32con
import pygetwindow as gw
import numpy as np
import cv2
from mss import mss

def capture_scum_window():
    """Capture only the SCUM window content (client area). Returns BGR image or None."""
    wins = [w for w in gw.getAllWindows() if w.title.strip() == "SCUM"]
    if not wins:
        return None

    win = wins[0]
    hwnd = win._hWnd

    # Get client rectangle (game area only, excludes title bar and borders)
    rect = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))

    width, height = right - left, bottom - top

    with mss() as sct:
        sct_img = sct.grab({"top": top, "left": left, "width": width, "height": height})
        img = np.array(sct_img)  # BGRA
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr

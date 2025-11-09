# 🧩 SCUM Panel Solver Capture

A tool to automatically capture and solve the **SCUM** in-game electrical panels  
using **screenshot recognition (OCR)** and an interactive solver.

---

## 🚀 Features

- 📷 **Automatic SCUM window capture**
- 🔍 **OCR recognition** (via Tesseract) for Input, Output A, Output B, and Slot values
- 🧠 **Solver** that finds the correct slot combination automatically
- 🎨 **Color-coded overlay:**
  - 🟢 Green = active slot (part of the solution)
  - 🔴 Red = inactive slot
- 🧰 **Manual editing**:
  - Operators (`+`, `-`, `*`, `/`)
  - Values (2, 10, 20, 40, 60, 80, 100)
- 🧩 **Real-time visual feedback** beside S1–S8 labels

---

## 🖥️ Installation

### 1️⃣ Requirements

- **Python 3.9+**
- **Tesseract OCR**

Download and install Tesseract OCR from here:  
👉 [Tesseract Windows Installer (UB Mannheim)](https://github.com/UB-Mannheim/tesseract/wiki)

After installation, add the executable path in your Python code if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

### 2️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
tk
opencv-python
pillow
pytesseract
numpy
```

---

## 🧩 Usage

1. **Launch SCUM** and open the panel you want to analyze.  
2. **Run the tool:**

   ```bash
   python main.py
   ```

3. Click **📷 Capture SCUM** to automatically grab the current SCUM window.  
   Or use **🖼 Load Image** to open a saved screenshot.

4. Press **🔍 Analyze** to detect input/output/slot values via OCR.  
5. Click **🧩 Solve Panel** to automatically compute the correct slot combination.  

   Color indicators (🟢/🔴) appear next to S1–S8 for the result.

---

## 🧭 Calibration Tool (calibrate_slots.py)

If your screen resolution or SCUM panel layout differs, use the included **calibration tool**:

```bash
python calibrate_slots.py
```

### Controls and Workflow

1. A file dialog will open – select a **screenshot** of the panel.  
2. The tool automatically detects the upper and lower reference screws.  
3. Use the following keys to define regions:  
   - **I** → Input ROI (green box)  
   - **A** → Output A ROI (red box)  
   - **B** → Output B ROI (blue box)  
   - **L** → Left operator area (blue rectangles beside slots)  
   - **R** → Right operator area (red rectangles beside slots)  
4. **Left-click and drag** to draw a region.  
5. **Right-click** anywhere to save → creates  
   - `calibration.json` (used by `main.py`)  
   - `calibration_overlay.png` (for visual check)  
6. Slot centers (S1–S8) can be moved by **drag & drop**.

> 💡 Tip: The help text also appears directly **inside the calibration tool** window.

---

## ⚙️ Calibration File Example

The `calibration.json` file defines the regions (ROIs) for OCR.  
You only need to calibrate once per resolution.

Example:

```json
{
  "input": [100, 200, 160, 240],
  "output_a": [200, 300, 260, 340],
  "output_b": [200, 400, 260, 440],
  "slots": [
    {
      "left_op": [100, 500, 160, 540],
      "right_op": [200, 500, 260, 540],
      "center": [180, 520]
    }
  ]
}
```

---

## 🧠 OCR Notes

- Supported operators and values:
  ```
  +, -, *, /
  2, 10, 20, 40, 60, 80, 100
  ```
- Common OCR misreads (`I`, `l`, `/`, `1`) are auto-corrected.

---

## 🧾 License

This project is **open-source** — free to use, modify, and extend.

---

## 👤 Author

**Stefan Kögl (HellBz)**  
📍 Karlsruhe, Germany  
💻 GitHub: [https://github.com/HellBz](https://github.com/HellBz)

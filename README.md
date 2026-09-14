# Image Processing Lab — Mini Project

A Flask + Python + OpenCV webpage that converts the 9 practicals from the supplied practical ZIP into browser-based operations.

## Student details
- Name: Shlok Kuthe
- USN: CS25D022
- Year: 3rd
- Semester: 5th
- Section: D

## Run on Windows / macOS / Linux

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).

### Windows PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

4. Open the address printed by Flask, normally `http://127.0.0.1:5000`.

## Notes
- Practical 5 is implemented exactly as a filter submenu: Box Filter, Gaussian Blur, Median Blur, Bilateral Filter, and All Blur Filters.
- Practical 9 takes two files: the input image and template image.
- Practical 6 can use the uploaded image; if no image is supplied for its synthetic-noise demonstrations, a small demo image is generated.
- The original practical ZIP contained a large `.venv` directory inside PR4. It is intentionally not copied into this project because it is environment-specific and not required to run the webpage.
- The webpage reproduces the operations represented by the submitted `.py` practical files rather than executing GUI commands such as `cv2.imshow()` on the server.

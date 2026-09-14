import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def read_image(file_storage, grayscale=False):
    data = file_storage.read()
    if not data:
        raise ValueError("No image data received.")
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flag)
    if img is None:
        raise ValueError("The uploaded file is not a valid image.")
    return img


def png_b64(img):
    if img is None:
        raise ValueError("Could not create output image.")
    if len(img.shape) == 2:
        ok, buf = cv2.imencode(".png", img)
    else:
        ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode output image.")
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def jpeg_bytes(img, quality=30):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG compression failed.")
    return buf.tobytes()


def png_bytes(img, compression=9):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise ValueError("PNG compression failed.")
    return buf.tobytes()


def make_card(title, img):
    return {"title": title, "image": png_b64(img)}


def base_gray_or_uploaded(files, key="image"):
    f = files.get(key)
    if not f:
        # A simple synthetic image makes the noise demonstrations runnable.
        base = np.ones((256, 256), dtype=np.uint8) * 127
        cv2.rectangle(base, (60, 60), (200, 200), 200, -1)
        cv2.putText(base, "IP LAB", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 50, 2)
        return base
    return read_image(f, grayscale=True)


def practical1(op, files):
    img = read_image(files["image"])
    return [make_card("Original Image", img)], "Image loaded successfully."


def practical2(op, files):
    if op in {"add", "subtract", "weighted_add", "and", "or", "xor"}:
        a = read_image(files["image1"])
        b = read_image(files["image2"])
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
        if op == "add":
            out = cv2.add(a, b)
        elif op == "subtract":
            out = cv2.subtract(a, b)
        elif op == "weighted_add":
            out = cv2.addWeighted(a, 0.5, b, 0.5, 0)
        elif op == "and":
            out = cv2.bitwise_and(b, a)
        elif op == "or":
            out = cv2.bitwise_or(b, a)
        else:
            out = cv2.bitwise_xor(b, a)
        return [make_card(op.replace("_", " ").title(), out)], "Operation completed."

    img = read_image(files["image"])
    if op == "color":
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif op == "grayscale":
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif op == "save":
        out = img
    elif op == "not":
        out = cv2.bitwise_not(img)
    elif op == "all":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cards = [
            make_card("Color Image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            make_card("Grayscale", gray),
            make_card("Saved/Loaded Image", img),
            make_card("Bitwise NOT", cv2.bitwise_not(img)),
        ]
        return cards, "All single-image PR2 operations completed. Pair operations need two uploaded images and are available separately."
    else:
        raise ValueError("Unknown PR2 operation.")
    return [make_card(op.title(), out)], "Operation completed."


def practical3(op, files):
    img = read_image(files["image"], grayscale=True)
    rows, cols = img.shape

    def translation():
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        return cv2.warpAffine(img, M, (cols, rows))

    def reflection():
        M = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
        return cv2.warpPerspective(img, M, (cols, rows))

    def rotation():
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 0.6)
        return cv2.warpAffine(img, M, (cols, rows))

    def resize():
        small = cv2.resize(img, (250, 200), interpolation=cv2.INTER_AREA)
        large = cv2.resize(small, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        return small, large

    def crop():
        y2, x2 = min(300, rows), min(300, cols)
        y1, x1 = min(100, max(0, y2 - 1)), min(100, max(0, x2 - 1))
        return img[y1:y2, x1:x2]

    def shear_x():
        M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        return cv2.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))

    def shear_y():
        M = np.float32([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
        return cv2.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))

    funcs = {
        "translation": translation,
        "reflection": reflection,
        "rotation": rotation,
        "crop": crop,
        "shear_x": shear_x,
        "shear_y": shear_y,
    }
    if op == "resize":
        s, l = resize()
        return [make_card("Shrinked", s), make_card("Enlarged", l)], "Resize completed."
    if op == "all":
        cards = [make_card("Original", img)]
        for key, title in [
            ("translation", "Translation"), ("reflection", "Reflection"),
            ("rotation", "Rotation"), ("crop", "Crop"),
            ("shear_x", "Shear X"), ("shear_y", "Shear Y")
        ]:
            cards.append(make_card(title, funcs[key]()))
        s, l = resize()
        cards += [make_card("Shrinked", s), make_card("Enlarged", l)]
        return cards, "All PR3 transformations completed."
    if op not in funcs:
        raise ValueError("Unknown PR3 operation.")
    return [make_card(op.replace("_", " ").title(), funcs[op]())], "Operation completed."


def practical4(op, files):
    img_color = read_image(files["image"])
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    def brightness1():
        return cv2.addWeighted(img_color, 2.3, np.zeros_like(img_color), 0, 10)

    def brightness2():
        return cv2.convertScaleAbs(img_color, alpha=1.5, beta=50)

    def negative():
        return 255 - img

    def median():
        return cv2.medianBlur(img_color, 11)

    def sharpen():
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img_color, -1, kernel)

    def laplacian():
        lap = cv2.Laplacian(img_color, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

    def equalize():
        return cv2.equalizeHist(img)

    def histogram():
        canvas = np.full((420, 900, 3), 255, np.uint8)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([equalize()], [0], None, [256], [0, 256]).flatten()
        m = max(float(hist.max()), float(hist2.max()), 1.0)
        for x in range(256):
            y1 = int(190 - hist[x] / m * 160)
            y2 = int(400 - hist2[x] / m * 160)
            cv2.line(canvas, (x, 190), (x, y1), (0, 0, 0), 1)
            cv2.line(canvas, (x + 450, 400), (x + 450, y2), (0, 0, 0), 1)
        cv2.putText(canvas, "Original Histogram", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2)
        cv2.putText(canvas, "Equalized Histogram", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2)
        return canvas

    def thresholds():
        names = [
            ("Binary", cv2.THRESH_BINARY), ("Binary Inverted", cv2.THRESH_BINARY_INV),
            ("Truncated", cv2.THRESH_TRUNC), ("To Zero", cv2.THRESH_TOZERO),
            ("To Zero Inverted", cv2.THRESH_TOZERO_INV)
        ]
        tiles = []
        for name, flag in names:
            _, t = cv2.threshold(img, 120, 255, flag)
            tile = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2)
            tiles.append(tile)
        h, w = img.shape
        blank = np.zeros((h, w, 3), np.uint8)
        top = cv2.hconcat([tiles[0], tiles[1]])
        mid = cv2.hconcat([tiles[2], tiles[3]])
        bot = cv2.hconcat([tiles[4], blank])
        return cv2.vconcat([top, mid, bot])

    funcs = {
        "negative": negative, "brightness_contrast_1": brightness1,
        "brightness_contrast_2": brightness2, "median": median,
        "sharpen": sharpen, "laplacian": laplacian,
        "equalize": equalize, "histogram": histogram, "thresholds": thresholds
    }
    if op == "all":
        cards = [make_card("Original", img_color)]
        for k, t in [
            ("negative","Negative"), ("brightness_contrast_1","Brightness & Contrast 1"),
            ("brightness_contrast_2","Brightness & Contrast 2"), ("median","Median Blur"),
            ("sharpen","Sharpening"), ("laplacian","Laplacian Sharpening"),
            ("equalize","Histogram Equalization"), ("histogram","Histogram"),
            ("thresholds","Thresholding")
        ]:
            cards.append(make_card(t, funcs[k]()))
        return cards, "All PR4 operations completed."
    return [make_card(op.replace("_"," ").title(), funcs[op]())], "Operation completed."


def practical5(op, files):
    img = read_image(files["image"])
    funcs = {
        "box_filter": lambda: cv2.boxFilter(img, -1, (2, 2), normalize=True),
        "gaussian_blur": lambda: cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT),
        "median_blur": lambda: cv2.medianBlur(img, 5),
        "bilateral_filter": lambda: cv2.bilateralFilter(img, 9, 75, 75),
    }
    if op == "all":
        return [make_card("Original", img)] + [
            make_card(t, funcs[k]()) for k,t in [
                ("box_filter","Box Filter"), ("gaussian_blur","Gaussian Blur"),
                ("median_blur","Median Blur"), ("bilateral_filter","Bilateral Filter")
            ]
        ], "All blur filters completed."
    return [make_card(op.replace("_"," ").title(), funcs[op]())], "Filter completed."


def practical6(op, files):
    img = base_gray_or_uploaded(files)
    def gaussian_noise():
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        return cv2.add(img.astype(np.int16), noise, dtype=cv2.CV_8U)
    def salt_pepper():
        out = img.copy()
        n = min(2000, img.size // 3)
        ys = np.random.randint(0, img.shape[0], n)
        xs = np.random.randint(0, img.shape[1], n)
        out[ys, xs] = 255
        ys = np.random.randint(0, img.shape[0], n)
        xs = np.random.randint(0, img.shape[1], n)
        out[ys, xs] = 0
        return out
    def general_noise():
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        return cv2.add(img, noise)
    def scratch():
        out = img.copy()
        cv2.line(out, (30, max(20,img.shape[0]//3)), (max(40,img.shape[1]-30), max(25,img.shape[0]//2)), 0, 3)
        cv2.line(out, (max(20,img.shape[1]//2-30), 20), (max(25,img.shape[1]//2+10), max(40,img.shape[0]-20)), 0, 3)
        return out

    funcs = {
        "generate_mask": lambda: cv2.threshold(img, 5, 255, cv2.THRESH_BINARY_INV)[1],
        "telea_inpainting": lambda: cv2.inpaint(img, cv2.threshold(img, 5, 255, cv2.THRESH_BINARY_INV)[1], 3, cv2.INPAINT_TELEA),
        "ns_inpainting": lambda: cv2.inpaint(img, cv2.threshold(img, 5, 255, cv2.THRESH_BINARY_INV)[1], 3, cv2.INPAINT_NS),
        "gaussian_noise": gaussian_noise,
        "gaussian_restore": lambda: cv2.GaussianBlur(img, (5,5), 0),
        "salt_pepper_noise": salt_pepper,
        "median_restore": lambda: cv2.medianBlur(img, 5),
        "general_noise": general_noise,
        "nlm_restore": lambda: cv2.fastNlMeansDenoising(img, None, 30, 7, 21),
        "scratched_image": scratch,
    }
    if op == "all":
        g = gaussian_noise()
        sp = salt_pepper()
        gn = general_noise()
        sc = scratch()
        mask = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY_INV)[1]
        return [
            make_card("Original", img),
            make_card("Generated Mask", mask),
            make_card("Telea Inpainting", cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)),
            make_card("Navier-Stokes Inpainting", cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)),
            make_card("Gaussian Noise", g),
            make_card("Gaussian Blur Restoration", cv2.GaussianBlur(g,(5,5),0)),
            make_card("Salt & Pepper Noise", sp),
            make_card("Median Restoration", cv2.medianBlur(sp,5)),
            make_card("General Noise", gn),
            make_card("Non-local Means Restoration", cv2.fastNlMeansDenoising(gn,None,30,7,21)),
            make_card("Scratched Image", sc),
        ], "All PR6 operations completed. Uploading your own image is optional for synthetic-noise demonstrations."
    return [make_card(op.replace("_"," ").title(), funcs[op]())], "Operation completed."


def rle_encode(data):
    enc=[]; prev=data[0]; count=1
    for p in data[1:]:
        if p == prev: count += 1
        else: enc.append((prev,count)); prev=p; count=1
    enc.append((prev,count)); return enc


def lzw_compress(data):
    dictionary={bytes([i]):i for i in range(256)}
    w=b""; out=[]; size=256
    for k in data:
        c=bytes([k]); wc=w+c
        if wc in dictionary: w=wc
        else:
            out.append(dictionary[w]); dictionary[wc]=size; size+=1; w=c
    if w: out.append(dictionary[w])
    return out


def practical7(op, files):
    img = read_image(files["image"])
    raw = img.size if img.ndim == 2 else img.shape[0]*img.shape[1]
    jpg = jpeg_bytes(img, 30)
    png = png_bytes(img, 9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten().tolist()
    rle = rle_encode(pixels)
    lzw = lzw_compress(pixels)
    rle_size = len(rle)*2
    lzw_size = len(lzw)*2
    metrics = {
        "Original image (KB)": round(len(cv2.imencode(".png", img)[1]) / 1024, 2),
        "JPEG lossy (KB)": round(len(jpg)/1024, 2),
        "PNG lossless (KB)": round(len(png)/1024, 2),
        "JPEG ratio": round(len(cv2.imencode(".png", img)[1]) / len(jpg), 2) if jpg else 0,
        "PNG ratio": round(len(cv2.imencode(".png", img)[1]) / len(png), 2) if png else 0,
        "RLE size (approx bytes)": rle_size,
        "LZW size (approx bytes)": lzw_size,
        "RLE ratio": round(len(pixels)/rle_size, 2) if rle_size else 0,
        "LZW ratio": round(len(pixels)/lzw_size, 2) if lzw_size else 0,
    }
    if op == "jpeg":
        return [make_card("Lossy JPEG (Quality 30)", cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR))], metrics
    if op == "png":
        return [make_card("Lossless PNG (Compression 9)", cv2.imdecode(np.frombuffer(png,np.uint8), cv2.IMREAD_COLOR))], metrics
    if op == "rle":
        return [make_card("Original Grayscale", gray)], metrics
    if op == "lzw":
        return [make_card("Original Grayscale", gray)], metrics
    if op == "all":
        return [
            make_card("Original", img),
            make_card("JPEG Lossy", cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)),
            make_card("PNG Lossless", cv2.imdecode(np.frombuffer(png,np.uint8), cv2.IMREAD_COLOR)),
            make_card("Grayscale for RLE/LZW", gray),
        ], metrics
    raise ValueError("Unknown PR7 operation.")


def practical8(op, files):
    img = read_image(files["image"], grayscale=True)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    funcs = {
        "erosion": lambda: cv2.erode(binary,kernel,iterations=1),
        "dilation": lambda: cv2.dilate(binary,kernel,iterations=1),
        "opening": lambda: cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel),
        "closing": lambda: cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel),
    }
    def analysis(opimg):
        contours,_=cv2.findContours(opimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        areas=[round(float(cv2.contourArea(c)),2) for c in contours]
        out=cv2.cvtColor(opimg,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(out,contours,-1,(0,0,255),2)
        return out, len(areas), areas
    if op == "object_analysis":
        out,n,areas=analysis(binary)
        return [make_card("Objects + Contours",out)], {"Number of objects":n,"Areas":areas}
    if op == "all":
        cards=[make_card("Original Binary",binary)]
        info={}
        for k,t in [("erosion","Erosion"),("dilation","Dilation"),("opening","Opening"),("closing","Closing")]:
            im=funcs[k](); out,n,areas=analysis(im)
            cards.append(make_card(t,out)); info[t]={"objects":n,"areas":areas}
        return cards, info
    return [make_card(op.title(), funcs[op]())], "Morphological operation completed."


def practical9(op, files):
    template = read_image(files["template"], grayscale=True)
    img = read_image(files["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = template.shape
    if h > gray.shape[0] or w > gray.shape[1]:
        raise ValueError("Template image must be smaller than the input image.")
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc=np.where(res>=0.7)
    out=img.copy()
    count=0
    for pt in zip(*loc[::-1]):
        cv2.rectangle(out,pt,(pt[0]+w,pt[1]+h),(0,255,255),2); count+=1
    return [make_card("Detected Objects",out)], f"Detected {count} matching location(s) using threshold 0.70."



def practical10(op, files):
    img = read_image(files["image"], grayscale=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    corrected = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    if op == "tophat":
        return [make_card("Top-Hat Transformation", corrected)], "Top-hat transformation completed."
    if op == "background":
        return [make_card("Estimated Uneven Background", background)], "Background estimated using morphological opening."
    if op == "all":
        return [make_card("Original Grayscale", img), make_card("Estimated Background", background),
                make_card("Top-Hat Enhanced Details", corrected)], "Top-hat illumination correction completed."


def practical11(op, files):
    img = read_image(files["image"])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if op == "rgb":
        out = rgb
    elif op == "hsv":
        out = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2RGB)
    elif op == "ycrcb":
        out = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), cv2.COLOR_YCrCb2RGB)
    elif op == "lab":
        out = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2Lab), cv2.COLOR_Lab2RGB)
    elif op == "all":
        return [make_card("RGB", rgb),
                make_card("HSV (visualized)", cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2RGB)),
                make_card("YCrCb (visualized)", cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), cv2.COLOR_YCrCb2RGB)),
                make_card("Lab (visualized)", cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2Lab), cv2.COLOR_Lab2RGB))], \
               "RGB, HSV, YCrCb and Lab conversions completed."
    else:
        raise ValueError("Unknown colour-space operation.")
    return [make_card(op.upper() if op != "ycrcb" else "YCrCb", out)], "Colour-space conversion completed."


def practical12(op, files):
    img = read_image(files["image"], grayscale=True)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 200)
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    px = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    py = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
    prewitt = cv2.convertScaleAbs(cv2.magnitude(cv2.filter2D(blur,cv2.CV_32F,px),
                                                 cv2.filter2D(blur,cv2.CV_32F,py)))
    if op == "all":
        return [make_card("Original", img), make_card("Canny", canny),
                make_card("Sobel", sobel), make_card("Prewitt", prewitt)], \
               "Canny, Sobel and Prewitt edge detection completed for comparison."
    out={"canny":canny,"sobel":sobel,"prewitt":prewitt}.get(op)
    if out is None: raise ValueError("Unknown edge detector.")
    return [make_card(op.title(), out)], f"{op.title()} edge detection completed."


OPS = {
    1: [("display","Read & Display Image", ["image"])],
    2: [
        ("color","Display Color Image",["image"]),("grayscale","Convert to Grayscale",["image"]),
        ("save","Save/Load Image",["image"]),("add","Image Addition",["image1","image2"]),
        ("subtract","Image Subtraction",["image1","image2"]),("weighted_add","Weighted Addition",["image1","image2"]),
        ("and","Bitwise AND",["image1","image2"]),("or","Bitwise OR",["image1","image2"]),
        ("xor","Bitwise XOR",["image1","image2"]),("not","Bitwise NOT",["image"]),
        ("all","All Single-Image Operations",["image"])
    ],
    3: [(k,t,["image"]) for k,t in [
        ("translation","Translation"),("reflection","Reflection"),("rotation","Rotation"),
        ("resize","Resize (Shrink + Enlarge)"),("crop","Crop"),("shear_x","Shear X"),
        ("shear_y","Shear Y"),("all","All Transformations")]],
    4: [(k,t,["image"]) for k,t in [
        ("negative","Negative Image"),("brightness_contrast_1","Brightness & Contrast 1"),
        ("brightness_contrast_2","Brightness & Contrast 2"),("median","Median Blur"),
        ("sharpen","Sharpening"),("laplacian","Laplacian Sharpening"),
        ("equalize","Histogram Equalization"),("histogram","Histogram"),
        ("thresholds","Thresholding"),("all","All Operations")]],
    5: [(k,t,["image"]) for k,t in [
        ("box_filter","Box Filter"),("gaussian_blur","Gaussian Blur"),
        ("median_blur","Median Blur"),("bilateral_filter","Bilateral Filter"),
        ("all","All Blur Filters")]],
    6: [(k,t,["image"]) for k,t in [
        ("generate_mask","Generate Mask"),("telea_inpainting","Telea Inpainting"),
        ("ns_inpainting","Navier-Stokes Inpainting"),("gaussian_noise","Gaussian Noise"),
        ("gaussian_restore","Gaussian Blur Restoration"),("salt_pepper_noise","Salt & Pepper Noise"),
        ("median_restore","Median Restoration"),("general_noise","General Noise"),
        ("nlm_restore","Non-local Means Restoration"),("scratched_image","Scratched Image"),
        ("all","All Restoration/Noise Operations")]],
    7: [("jpeg","JPEG Lossy Compression",["image"]),("png","PNG Lossless Compression",["image"]),
        ("rle","RLE Compression Analysis",["image"]),("lzw","LZW Compression Analysis",["image"]),
        ("all","All Compression Operations",["image"])],
    8: [("erosion","Erosion",["image"]),("dilation","Dilation",["image"]),
        ("opening","Opening",["image"]),("closing","Closing",["image"]),
        ("object_analysis","Object Area + Contours",["image"]),("all","All Morphology Operations",["image"])],
    9: [("detect","Template Matching / Object Detection",["image","template"])],
    10: [("tophat","Top-Hat Transformation",["image"]),("background","Estimate Uneven Background",["image"]),("all","All Illumination Operations",["image"])],
    11: [("rgb","RGB Colour Space",["image"]),("hsv","HSV Colour Space",["image"]),("ycrcb","YCrCb Colour Space",["image"]),("lab","Lab Colour Space",["image"]),("all","All Colour Spaces",["image"])],
    12: [("canny","Canny Edge Detector",["image"]),("sobel","Sobel Edge Detector",["image"]),("prewitt","Prewitt Edge Detector",["image"]),("all","Compare All Edge Detectors",["image"])]
}


def run_operation(pr, op, files):
    if pr == 1: return practical1(op,files)
    if pr == 2: return practical2(op,files)
    if pr == 3: return practical3(op,files)
    if pr == 4: return practical4(op,files)
    if pr == 5: return practical5(op,files)
    if pr == 6: return practical6(op,files)
    if pr == 7: return practical7(op,files)
    if pr == 8: return practical8(op,files)
    if pr == 9: return practical9(op,files)
    if pr == 10: return practical10(op,files)
    if pr == 11: return practical11(op,files)
    if pr == 12: return practical12(op,files)
    raise ValueError("Invalid practical.")


@app.get("/")
def index():
    return render_template("index.html", ops=OPS)


@app.post("/api/run/<int:pr>/<op>")
def api_run(pr, op):
    try:
        spec = next((x for x in OPS.get(pr, []) if x[0] == op), None)
        if not spec:
            raise ValueError("Unknown operation.")
        missing = [k for k in spec[2] if k not in request.files or not request.files[k].filename]
        # PR6 can run without an upload, so do not require its image.
        if not (pr == 6 and missing == ["image"]) and missing:
            raise ValueError("Please upload: " + ", ".join(missing))
        cards, info = run_operation(pr, op, request.files)
        return jsonify({"ok": True, "cards": cards, "info": info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get("/api/ops/<int:pr>")
def api_ops(pr):
    return jsonify([{"id":a,"label":b,"inputs":c} for a,b,c in OPS.get(pr,[])])


if __name__ == "__main__":
    app.run(debug=True)

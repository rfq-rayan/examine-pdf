"""Shared helpers for the Prothom Alo e-paper extractor: page rendering, layout detection, OCR."""
import csv
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA = str(HERE / "tessdata")          # tessdata_best ben + eng
LAYOUT_MODEL = HERE / "models" / "doclayout.pt"  # DocLayout-YOLO (DocStructBench)


def extract_page_images(pdf_path, out_dir):
    """Dump the original page JPEGs (no re-encode) from the PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        p = out_dir / f"p{i + 1:02d}.jpg"
        if not p.exists():
            imgs = page.get_images()
            if len(imgs) == 1 and doc.xref_get_key(imgs[0][0], "Filter")[1] == "/DCTDecode":
                p.write_bytes(doc.xref_stream_raw(imgs[0][0]))
            else:
                page.get_pixmap(dpi=100).save(str(p))
        paths.append(p)
    return paths


_model = None


def layout_boxes(img_path, cache_path=None, imgsz=1600, conf=0.2):
    """DocLayout-YOLO regions for one page: list of dict(cls, conf, box=[x0,y0,x1,y1])."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text(encoding="utf8"))
    global _model
    if _model is None:
        from doclayout_yolo import YOLOv10
        _model = YOLOv10(str(LAYOUT_MODEL))
    r = _model.predict(str(img_path), imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    out = [dict(cls=r.names[int(c)], conf=round(cf, 3), box=[round(v, 1) for v in b])
           for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    if cache_path:
        Path(cache_path).write_text(json.dumps(out), encoding="utf8")
    return out


def _tess(img, psm, lang="ben"):
    ok, buf = cv2.imencode(".png", img)
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    r = subprocess.run([TESS, "stdin", "stdout", "-l", lang, "--tessdata-dir", TESSDATA, "--psm", str(psm), "tsv"],
                       input=buf.tobytes(), capture_output=True, env=env)
    rows = list(csv.reader(io.StringIO(r.stdout.decode("utf8", "replace")), delimiter="\t", quoting=csv.QUOTE_NONE))
    lines = {}
    for row in rows[1:]:
        if len(row) < 12 or not row[11].strip():
            continue
        try:
            conf = float(row[10])
        except ValueError:
            continue
        if conf < 0:
            continue
        key = (row[2], row[3], row[4])
        L, T, W, H = map(int, row[6:10])
        ln = lines.setdefault(key, dict(words=[], x0=L, y0=T, x1=L + W, y1=T + H, conf=[]))
        ln["words"].append(row[11])
        ln["x0"] = min(ln["x0"], L); ln["y0"] = min(ln["y0"], T)
        ln["x1"] = max(ln["x1"], L + W); ln["y1"] = max(ln["y1"], T + H)
        ln["conf"].append(conf)
    out = []
    for ln in lines.values():
        out.append(dict(text=" ".join(ln["words"]), x0=ln["x0"], y0=ln["y0"], x1=ln["x1"], y1=ln["y1"],
                        conf=sum(ln["conf"]) / len(ln["conf"])))
    out.sort(key=lambda d: (d["y0"] + d["y1"]) / 2)
    return out


def ocr_region(page_img, box, pad=8, psm=6):
    """OCR a crop of the page; returns lines with page-coordinate boxes."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = page_img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = page_img[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return []
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # white border helps tesseract on tight crops
    g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    lines = _tess(g, psm)
    for ln in lines:
        ln["x0"] += cx0 - 12; ln["x1"] += cx0 - 12
        ln["y0"] += cy0 - 12; ln["y1"] += cy0 - 12
    return lines


def pmap(fn, items, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def ocr_title(page_img, box, pad=8):
    """Headlines: OCR at several scales, keep the line-by-line best-confidence reading."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = page_img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    g = cv2.cvtColor(page_img[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
    if g.size == 0:
        return []
    g = cv2.copyMakeBorder(g, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    runs = []
    for sc in (1.0, 0.7, 0.5):
        c = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc != 1 else g
        ls = _tess(c, 6)
        for ln in ls:
            for k in ("x0", "x1", "y0", "y1"):
                ln[k] = ln[k] / sc
        runs.append(ls)
    runs = [r for r in runs if r]
    if not runs:
        return []
    counts = [len(r) for r in runs]
    modal = max(set(counts), key=counts.count)
    cand = [r for r in runs if len(r) == modal]
    out = []
    for i in range(modal):
        best = max((r[i] for r in cand), key=lambda l: l["conf"])
        out.append(dict(best))
    for ln in out:
        ln["x0"] += cx0 - 15; ln["x1"] += cx0 - 15
        ln["y0"] += cy0 - 15; ln["y1"] += cy0 - 15
    return out

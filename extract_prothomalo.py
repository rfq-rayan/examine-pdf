#!/usr/bin/env python3
"""Article extractor for the Prothom Alo e-paper PDF (raster pages + weak OCR layer).

The PDF only holds one JPEG per page, so everything is rebuilt from pixels:

  1. Dump the original page JPEGs from the PDF.
  2. Detect layout regions (headline / text block / photo / caption) with DocLayout-YOLO.
  3. Re-OCR every region with Tesseract `tessdata_best` Bengali (the PDF's own OCR layer is
     only used for nothing -- it is noticeably worse than a per-region OCR of the pixels).
  4. Classify headline-like boxes (headline / kicker / deck / byline / "continued" tag /
     in-body sub-head) from OCR text and geometry.
  5. Give every text block / photo to the nearest headline above it that covers its columns.
  6. Parse "এরপর পৃষ্ঠা N কলাম M" jump lines and merge them with the "প্রথম পৃষ্ঠার পর"
     continuations (page + column + headline similarity).
  7. Mask the redistributor's branding (Daily ePapers logo / footer) and drop ads.

Usage:
    python extract_prothomalo.py "Prothom Alo 06-10-2026 Daily_ePapers_BD.pdf" [out_dir]
"""
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from pa_common import extract_page_images, layout_boxes, ocr_region, ocr_title, pmap

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BN_RE = re.compile(r"[ঀ-৿]")
JUMP_OUT_RE = re.compile(r"এরপর\s*পৃষ্ঠা\s*([০-৯0-9]+)\s*কলাম\s*([০-৯0-9]+)")
JUMP_IN_RE = re.compile(r"(প্রথম\s*পৃষ্ঠার|পৃষ্ঠা\s*[০-৯0-9]+\s*এর|পৃষ্ঠার)\s*পর")
BYLINE_RE = re.compile(
    r"(প্রতিবেদক|প্রতিনিধি|সংবাদদাতা|প্রতিবেদন|ডেস্ক|বার্তা\s*সংস্থা|সংবাদ\s*সম্মেলন$|"
    r"রয়টার্স|এএফপি|এপি|আল-জাজিরা|বিবিসি|সিএনএন)")
CAPTION_CREDIT_RE = re.compile(r"(ছবি|ছবিঃ|ছবি:|গ্রাফিক|ফাইল\s*ছবি)")
STOP = set()

# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
@dataclass(eq=False)
class El:
    page: int
    cls: str            # title | text | figure | caption | other
    box: list           # x0,y0,x1,y1 in page px
    conf: float = 0.0
    lines: list = field(default_factory=list)   # OCR lines (page coords)
    role: str = ""      # headline | kicker | deck | byline | tag | subhead | body | photo | caption
    cols: tuple = ()
    owner: object = None
    text: str = ""
    skip: str = ""      # reason it is excluded (branding / header / ad ...)

    @property
    def x0(self): return self.box[0]
    @property
    def y0(self): return self.box[1]
    @property
    def x1(self): return self.box[2]
    @property
    def y1(self): return self.box[3]
    @property
    def w(self): return self.box[2] - self.box[0]
    @property
    def h(self): return self.box[3] - self.box[1]


@dataclass(eq=False)
class Article:
    page: int
    head: El
    kickers: list = field(default_factory=list)
    decks: list = field(default_factory=list)
    bylines: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    blocks: list = field(default_factory=list)   # body El (text/subhead), unordered
    photos: list = field(default_factory=list)   # (figure El, caption El or None)
    notes: list = field(default_factory=list)
    id: str = ""
    parts: list = field(default_factory=list)    # continuation Articles merged into this one
    jump_out: object = None                      # (page, col)
    section: str = "Main"
    label: str = ""

    @property
    def box(self):
        els = [self.head] + self.kickers + self.decks + self.bylines + self.blocks
        return [min(e.x0 for e in els), min(e.y0 for e in els), max(e.x1 for e in els), max(e.y1 for e in els)]


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def bn2int(s):
    return int(s.translate(BN_DIGITS))


def clean(t):
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


def bn_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(1 for c in letters if "ঀ" <= c <= "৿") / len(letters) if letters else 0.0


def ngrams(s, n=3):
    s = re.sub(r"[^ঀ-৿]", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def sim(a, b):
    A, B = ngrams(a), ngrams(b)
    return len(A & B) / max(1, min(len(A), len(B)))


def x_overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def y_overlap(a, b):
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def find_grid(els, page_w):
    """Column left edges from the x0 histogram of text blocks."""
    xs = sorted(e.x0 for e in els if e.cls == "text" and 250 <= e.w <= 450)
    peaks = []
    for x in xs:
        if peaks and x - peaks[-1][-1] < 45:
            peaks[-1].append(x)
        else:
            peaks.append([x])
    peaks = [p for p in peaks if len(p) >= 3]
    lefts = [float(np.median(p)) for p in peaks]
    if len(lefts) < 3:
        step = (page_w - 200) / 6
        lefts = [100 + i * step for i in range(6)]
    cw = float(np.median([e.w for e in els if e.cls == "text" and 250 <= e.w <= 450] or [370]))
    return lefts, cw


def cols_of(box, lefts, cw):
    out = []
    for i, l in enumerate(lefts):
        ov = x_overlap(box, [l, 0, l + cw, 0])
        if ov >= 0.35 * min(cw, box[2] - box[0]):
            out.append(i)
    return tuple(out) or (min(range(len(lefts)), key=lambda i: abs(lefts[i] - box[0])),)


# ---------------------------------------------------------------------------------------
# per-page processing
# ---------------------------------------------------------------------------------------
def load_elements(pg, img, layout):
    H, W = img.shape[:2]
    els = []
    cmap = {"title": "title", "plain text": "text", "figure": "figure", "figure_caption": "caption",
            "table": "text", "table_caption": "caption", "table_footnote": "text"}
    for d in layout:
        cls = cmap.get(d["cls"])
        if not cls:
            continue
        x0, y0, x1, y1 = d["box"]
        el = El(pg, cls, [max(0, x0), max(0, y0), min(W, x1), min(H, y1)], d["conf"])
        el.k = len(els)
        els.append(el)
    # drop duplicate detections (same class, one box mostly inside the other)
    els.sort(key=lambda e: -e.conf)
    kept = []
    for e in els:
        dup = False
        for k in kept:
            if k.cls == e.cls:
                inter = x_overlap(e.box, k.box) * y_overlap(e.box, k.box)
                if inter > 0.6 * min(e.w * e.h, k.w * k.h):
                    dup = True
                    break
        if not dup:
            kept.append(e)
    kept.sort(key=lambda e: (e.y0, e.x0))
    return kept


def mask_branding(els, pg, W, H):
    for e in els:
        # footer ribbon ("Daily ePapers" with the Telegram link) sits in the bottom ~90 px
        if e.y0 > H - 120:
            e.skip = "branding"
        # logo box on page 1 top-left
        if pg == 1 and e.y1 < 520 and e.x0 < 760:
            e.skip = "branding"


def ocr_all(pages, work):
    """Tesseract every non-photo region; results cached per page in work/ocr_pNN.json."""
    jobs = []
    for pg, img, els in pages:
        cache = work / f"ocr2_p{pg:02d}.json"
        data = json.loads(cache.read_text(encoding="utf8")) if cache.exists() else {}
        for e in els:
            k = e.k
            if e.skip or e.cls == "figure":
                continue
            key = ("t" if e.cls == "title" else "") + str(k)
            if key in data:
                e.lines = data[key]
                e.text = clean(chr(10).join(l["text"] for l in e.lines))
            else:
                jobs.append((pg, k, img, e))

    def run(job):
        pg, k, img, e = job
        e.lines = ocr_title(img, e.box) if e.cls == 'title' else ocr_region(img, e.box, pad=6, psm=6)
        e.text = clean(chr(10).join(l["text"] for l in e.lines))
    pmap(run, jobs, workers=7)
    for pg, img, els in pages:
        cache = work / f"ocr2_p{pg:02d}.json"
        cache.write_text(json.dumps({("t" if e.cls == "title" else "") + str(e.k): e.lines for e in els if e.lines}, ensure_ascii=False),
                         encoding="utf8")


def drop_garbage(els):
    for e in els:
        if e.skip or e.cls == "figure":
            continue
        letters = sum(c.isalpha() for c in e.text)
        if not e.lines or letters < 2:
            e.skip = "empty"
            continue
        # OCR of a photo or logo area -> nonsense
        conf = sum(l["conf"] for l in e.lines) / len(e.lines)
        if bn_ratio(e.text) < 0.5 and not re.search(r"[A-Za-z]{4}", e.text):
            e.skip = "garbage"


def line_height(e):
    ls = e.lines
    if not ls:
        return e.h
    return float(np.median([l["y1"] - l["y0"] for l in ls]))


def classify_titles(els, lefts, cw, W):
    """Assign roles to title boxes; returns list of headline clusters."""
    titles = sorted([e for e in els if e.cls == "title" and not e.skip], key=lambda e: (e.y0, e.x0))
    texts = [e for e in els if e.cls == "text" and not e.skip]
    for t in titles:
        t.lh = line_height(t)
        t.cols = cols_of(t.box, lefts, cw)
        t.nlines = max(1, len(t.lines))
    # --- pass 1: text-pattern roles
    for t in titles:
        s = t.text.replace("\n", " ")
        if JUMP_IN_RE.search(s) and len(s) < 40:
            t.role = "tag"
        elif len(s) < 70 and t.nlines <= 2 and BYLINE_RE.search(s) and t.h < 80:
            t.role = "byline"
    # --- pass 2: in-body sub-heads: small title directly below a text block in the same columns
    figs = [e for e in els if e.cls == "figure" and not e.skip]
    for t in titles:
        if t.role:
            continue
        # a small title right under a photo is its caption
        if t.h < 140 and t.lh < 36 and any(-10 < t.y0 - f.y1 < 60 and x_overlap(f.box, t.box) > 0.5 * t.w for f in figs):
            t.role = "caption"
            continue
        below_big = any(u is not t and not u.role and u.lh > 1.2 * t.lh and 0 <= u.y0 - t.y1 < 75
                        and x_overlap(u.box, t.box) > 0.3 * min(u.w, t.w) for u in titles)
        if below_big:
            continue
        above = [b for b in texts if b.y1 <= t.y0 + 8 and t.y0 - b.y1 < 70
                 and x_overlap(b.box, t.box) > 0.6 * min(b.w, t.w)]
        if t.h < 85 and t.lh < 40 and above:
            t.role = "subhead"
    # --- pass 3: headline candidates and clusters
    heads = [t for t in titles if not t.role]
    # merge neighbouring big boxes that are really one headline (split by the detector)
    heads.sort(key=lambda e: (e.y0, e.x0))
    merged = []
    for t in heads:
        for m in merged:
            same_row = (abs(m.lh - t.lh) < 0.22 * max(m.lh, t.lh) and
                        y_overlap(m.box, t.box) > 0.7 * min(m.h, t.h) and 0 <= t.x0 - m.x1 < 110)
            if same_row or (abs(m.lh - t.lh) < 0.22 * max(m.lh, t.lh) and -15 < t.y0 - m.y1 < 0.9 * t.lh
                            and x_overlap(m.box, t.box) > 0.3 * min(m.w, t.w)):
                m.box = [min(m.x0, t.x0), min(m.y0, t.y0), max(m.x1, t.x1), max(m.y1, t.y1)]
                m.lines += t.lines
                m.text += "\n" + t.text
                m.nlines += t.nlines
                m.cols = cols_of(m.box, lefts, cw)
                t.skip = "merged"
                break
        else:
            merged.append(t)
    heads = merged
    # kicker / deck: small heads that touch a bigger head
    big = sorted(heads, key=lambda e: -e.lh)
    clusters = []
    used = set()
    for t in big:
        if id(t) in used:
            continue
        near_big = [m for m in clusters
                    if m.head.lh > 1.25 * t.lh and x_overlap(m.head.box, t.box) > 0.3 * min(m.head.w, t.w)
                    and (-10 < t.y0 - m.head.y1 < 70 or -10 < m.head.y0 - t.y1 < 70)]
        if near_big and t.h < 130:
            m = near_big[0]
            if t.y1 <= m.head.y0 + 10:
                t.role = "kicker"; m.kickers.append(t)
            else:
                t.role = "deck"; m.decks.append(t)
            used.add(id(t))
        else:
            t.role = "headline"
            clusters.append(Article(t.page, t))
    # attach byline / tag to nearest cluster above (or just below) in the same columns
    for t in titles:
        if t.role in ("byline", "tag"):
            cand = [c for c in clusters if x_overlap(c.head.box, t.box) > 0.3 * min(c.head.w, t.w) and
                    -30 < t.y0 - c.head.y1 < 400]
            if not cand:
                # byline belonging to a headline in an adjacent column span
                cand = [c for c in clusters if x_overlap(c.head.box, t.box) > 0 and -30 < t.y0 - c.head.y1 < 500]
            if cand:
                c = min(cand, key=lambda c: abs(t.y0 - c.head.y1))
                (c.bylines if t.role == "byline" else c.tags).append(t)
            else:
                t.skip = "orphan-" + t.role
    for t in titles:
        if t.role == "caption":
            t.cls = "caption"
    return clusters, [t for t in titles if t.role == "subhead"]


def set_spans(clusters, lefts, cw):
    for c in clusters:
        cs = c.head.cols
        c.span = [min(c.head.x0, lefts[min(cs)]), 0, max(c.head.x1, lefts[max(cs)] + cw), 0]


def assign_blocks(clusters, els, subheads):
    """Hang text blocks, sub-heads, photos and captions on the nearest headline above them."""
    if not clusters:
        return
    def anchor_bottom(c):
        return max([c.head.y1] + [e.y1 for e in c.kickers + c.decks + c.bylines + c.tags])

    def cand_for(e, slack=14):
        out = []
        for c in clusters:
            hb = anchor_bottom(c)
            ov = x_overlap(e.box, c.span) / max(1, e.w)
            if ov >= 0.55 and hb <= e.y0 + slack:
                out.append((hb, c))
        return out

    for e in sorted([x for x in els if x.cls == "text" and not x.skip] + subheads, key=lambda x: x.y0):
        cs = cand_for(e)
        if not cs:
            # body column sitting beside a headline that is narrower than its text columns
            near = []
            for c in clusters:
                hb = anchor_bottom(c)
                gap = e.x0 - c.head.x1
                if -40 < gap < 130 and c.head.y0 - 20 <= e.y0 and e.y0 - c.head.y0 < 900 and                         not any(o is not c and o.head.y1 < e.y0 and o.head.y1 > c.head.y0 and
                                x_overlap(o.head.box, e.box) > 0.5 * e.w for o in clusters):
                    near.append((e.y0 - c.head.y0, c))
            if near:
                cs = [(0, min(near, key=lambda t: t[0])[1])]
        if not cs:
            e.skip = "unowned"
            continue
        c = max(cs, key=lambda t: t[0])[1]
        e.owner = c
        e.role = e.role or "body"
        c.blocks.append(e)


BLOCK_AD_RE = re.compile(r"(নিলাম বিজ্ঞপ্তি|টেন্ডার|দরপত্র|e-?Tender|e-?GP|https?://|www\.|\bPLC\b|\bLtd\b|Tel:|Fax|"
                         r"সংবাদ বিজ্ঞপ্তি|নিয়োগ বিজ্ঞপ্তি|পলাতক আসামি|শুভ জন্মদিন|আমরা গভীরভাবে)", re.I)


def latin_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(1 for c in letters if c.isascii()) / len(letters) if letters else 0.0


def cut_ads(clusters):
    """Blocks that read like ads/notices end their article: they and everything below them in
    the same columns are dropped (ads are stacked under the news at the page bottom)."""
    for a in clusters:
        ad_blocks = [b for b in a.blocks if b.role != "subhead" and
                     sum(c.isalpha() for c in b.text) >= 60 and
                     (len(BLOCK_AD_RE.findall(b.text)) >= 2 or latin_ratio(b.text) > 0.45)]
        if not ad_blocks:
            continue
        keep = []
        for b in a.blocks:
            cut = any(b is x or (b.y0 >= x.y0 - 5 and x_overlap(b.box, x.box) > 0.5 * min(b.w, x.w)) for x in ad_blocks)
            if cut:
                b.skip = "ad"
            else:
                keep.append(b)
        a.blocks = keep


def propagate_owners(clusters, els, subheads):
    """Blocks outside a headline's own column span: attach by vertical adjacency or column flow."""
    pool = [x for x in els if x.cls == "text" and x.skip in ("", "unowned") and x.owner is None] +            [x for x in subheads if x.owner is None]
    changed = True
    while pool and changed:
        changed = False
        for e in sorted(pool, key=lambda b: (b.y0, b.x0)):
            best = None
            for c in clusters:
                for b in c.blocks:
                    # same column, directly below an owned block
                    if abs(e.x0 - b.x0) < 40 and 0 <= e.y0 - b.y1 < 800 and x_overlap(e.box, b.box) > 0.6 * min(e.w, b.w)                             and not any(o.head.y0 > b.y1 - 10 and o.head.y1 < e.y0 + 14 and
                                        x_overlap(o.head.box, e.box) > 0.3 * e.w for o in clusters):
                        d = e.y0 - b.y1
                    # next column of the flow: just right of an owned block, starting near the top
                    elif 0 < e.x0 - b.x1 < 110 and e.y0 <= b.y0 + 200 and e.y0 >= c.head.y0 - 30 and                             not any(o is not c and o.head.y0 < e.y0 and o.head.y1 > c.head.y0 and
                                    x_overlap(o.head.box, e.box) > 0.5 * e.w and o.head.y1 <= e.y0 + 14 for o in clusters):
                        d = 150 + abs(e.y0 - b.y0)
                    else:
                        continue
                    if best is None or d < best[0]:
                        best = (d, c)
            if best is not None:
                e.owner = best[1]; e.skip = ""
                e.role = e.role or "body"
                best[1].blocks.append(e)
                pool.remove(e)
                changed = True
                break
    for e in els:
        if e.skip == "unowned" and e.owner is not None:
            e.skip = ""


def attach_photos(clusters, els):
    figs = [e for e in els if e.cls == "figure" and not e.skip]
    caps = [e for e in els if e.cls == "caption" and not e.skip]
    for f in figs:
        # a small caption directly below the photo
        cap = None
        best = 1e9
        for c in caps:
            if c.owner is not None:
                continue
            gap = c.y0 - f.y1
            if -20 < gap < 120 and x_overlap(f.box, c.box) > 0.4 * min(f.w, c.w) and gap < best:
                cap, best = c, gap
        f.cap = cap
        if cap:
            cap.owner = True
    return figs


def owner_for_photo(f, clusters):
    """Photo belongs to the nearest headline above (same columns) unless a headline sits right below it."""
    box = f.box
    above, below = None, None
    for c in clusters:
        span = [c.head.x0, 0, c.head.x1, 0]
        ov = x_overlap(box, span) / max(1, f.w)
        ov2 = x_overlap(span, box) / max(1, c.head.w)
        if ov < 0.25 and ov2 < 0.25:
            continue
        hb = max([c.head.y1] + [e.y1 for e in c.kickers + c.decks + c.bylines + c.tags])
        if hb <= f.y0 + 14:
            d = f.y0 - hb
            if above is None or d < above[0]:
                above = (d, c)
        elif c.head.y0 >= f.y1 - 14:
            d = c.head.y0 - (f.cap.y1 if getattr(f, "cap", None) else f.y1)
            if below is None or d < below[0]:
                below = (d, c)
    if above and below:
        # photo sits between two stories: prefer the one it is closer to; ties -> below (photo opens a story)
        return below[1] if below[0] + 40 < above[0] else above[1]
    return (above or below or (0, None))[1]


# ---------------------------------------------------------------------------------------
# body text assembly
# ---------------------------------------------------------------------------------------
def block_paragraphs(e, col_left, cw):
    """Join OCR lines of one text block into paragraphs using indentation / short last lines."""
    paras, cur = [], []
    prev = None
    lh = line_height(e)
    for ln in e.lines:
        t = ln["text"].strip()
        t = re.sub(r"\s*[_|]+\s*$", "", t)
        if not t:
            continue
        if JUMP_OUT_RE.search(t):
            cur.append(t)           # keep the jump line (parsed later) but nothing after it belongs to this story
            break
        indent = ln["x0"] - e.x0
        newp = False
        if prev is not None:
            short_prev = (e.x1 - prev["x1"]) > 0.16 * e.w
            if indent > 0.7 * lh or short_prev:
                newp = True
        if newp and cur:
            paras.append(cur); cur = []
        cur.append(t)
        prev = ln
    if cur:
        paras.append(cur)
    out = []
    for p in paras:
        s = p[0]
        for t in p[1:]:
            s = s + t if s.endswith(("-", "‐", "‑")) else s + " " + t
        out.append(clean(s))
    return out


def article_body(a, lefts, cw):
    """Ordered list of (type, text) from the article's blocks (column-major)."""
    blocks = [b for b in a.blocks if not b.skip]
    # nothing below a "এরপর পৃষ্ঠা N কলাম M" line in its column belongs to the story (teaser boxes, next kicker...)
    jumps = [b for b in blocks if JUMP_OUT_RE.search(b.text)]
    for b in list(blocks):
        if any(b is not jb and b.y0 >= jb.y1 - 5 and x_overlap(b.box, jb.box) > 0.5 * min(b.w, jb.w) for jb in jumps):
            b.skip = "after-jump"
            blocks.remove(b)
    blocks.sort(key=lambda b: (min(b.cols or (0,)), b.y0))
    out = []
    for b in blocks:
        if b.role == "subhead":
            out.append(("h", clean(b.text.replace("\n", " "))))
        else:
            for p in block_paragraphs(b, b.x0, cw):
                out.append(("p", p))
    return out


# ---------------------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------------------
def process(pdf_path, out_dir):
    out_dir = Path(out_dir)
    work = out_dir / "_work"
    paths = extract_page_images(pdf_path, work / "pages")
    pages = []
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        lay = layout_boxes(p, work / f"layout_{p.stem}.json")
        els = load_elements(i, img, lay)
        H, W = img.shape[:2]
        mask_branding(els, i, W, H)
        pages.append((i, img, els))
    # page header / masthead band: not article content
    for i, img, els in pages:
        H, W = img.shape[:2]
        band = 700 if i == 1 else 300
        for e in els:
            if e.y1 < band and not e.skip:
                e.skip = "header"
    ocr_all(pages, work)
    for i, img, els in pages:
        drop_garbage(els)
    return pages


# ---------------------------------------------------------------------------------------
# page -> articles
# ---------------------------------------------------------------------------------------
def build_page(pg, img, els):
    H, W = img.shape[:2]
    live = [e for e in els if not e.skip]
    lefts, cw = find_grid(live, W)
    for e in els:
        if e.cls in ("text", "figure", "caption"):
            e.cols = cols_of(e.box, lefts, cw)
    clusters, subheads = classify_titles(els, lefts, cw, W)
    set_spans(clusters, lefts, cw)
    assign_blocks(clusters, els, subheads)
    propagate_owners(clusters, els, subheads)
    cut_ads(clusters)
    clusters = merge_author_tags(clusters)
    figs = attach_photos(clusters, els)
    for f in figs:
        c = owner_for_photo(f, clusters)
        if c is not None:
            c.photos.append((f, f.cap))
    for a in clusters:
        a.page = pg
    return clusters, lefts, cw


def merge_author_tags(clusters):
    """Opinion pages: an author name set as its own tiny 'headline' just below the real headline."""
    out = []
    for c in sorted(clusters, key=lambda c: c.head.y0):
        t = clean(c.head.text.replace(chr(10), " "))
        if not c.blocks and len(t) <= 40 and len(t.split()) <= 5 and not c.kickers and not c.decks:
            host = [o for o in out if o.head.lh > 1.4 * c.head.lh and 0 < c.head.y0 - o.head.y1 < 320
                    and x_overlap(o.head.box, c.head.box) > 0.3 * c.head.w]
            if host:
                host[-1].bylines.append(c.head)
                continue
        out.append(c)
    return out


def head_text(a):
    return clean(a.head.text.replace(chr(10), " "))


def finalize_body(a, cw):
    """Body items + jump-out extraction."""
    items = article_body(a, None, cw)
    a.byline_extra = ""
    if items and items[0][0] == "p" and len(items[0][1]) < 50 and BYLINE_RE.search(items[0][1])             and not a.bylines:
        a.byline_extra = items[0][1]
        items = items[1:]
    out = []
    for typ, text in items:
        m = JUMP_OUT_RE.search(text)
        if m:
            a.jump_out = (bn2int(m.group(1)), bn2int(m.group(2)))
            rest = clean(JUMP_OUT_RE.sub("", text))
            if len(rest) > 3:
                out.append((typ, rest))
            continue
        out.append((typ, text))
    return out


def page_section(pg, width):
    return "Naksha" if width > 3000 else "Main"


def link_jumps(arts, page_info):
    """Merge 'এরপর পৃষ্ঠা N কলাম M' articles with their 'প্রথম পৃষ্ঠার পর' continuations."""
    conts = [a for a in arts if a.tags]
    used = set()
    for a in arts:
        a.cont = None
    # process in page order so chains (p1 -> p2 -> p4) resolve
    for a in sorted([x for x in arts if x.jump_out], key=lambda x: x.page):
        n, col = a.jump_out
        sec = a.section
        best, bs = None, 0.0
        for c in conts:
            if id(c) in used or c is a or c.section != sec or c.label != str(n):
                continue
            colmatch = (col - 1) in c.head.cols or (col - 1) in [min(b.cols) for b in c.blocks if b.cols]
            s = sim(head_text(a), head_text(c))
            score = (1.0 if colmatch else 0.0) + s
            if (colmatch or s > 0.5) and score > bs:
                best, bs = c, score
        if best is not None:
            used.add(id(best))
            a.cont = best
    return used


def chain(a):
    seq = [a]
    while getattr(seq[-1], "cont", None) is not None and len(seq) < 8:
        seq.append(seq[-1].cont)
    return seq


def edit1(a, b):
    """True if Levenshtein distance(a, b) <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = 0
    while i < min(la, lb) and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1:] == b[i + 1:]
    if la > lb:
        return a[i + 1:] == b[i:]
    return a[i:] == b[i + 1:]


WORD_RE = re.compile(r"[ঀ-৿]+")


def build_vocab(pages):
    from collections import Counter
    c = Counter()
    for pg, img, els in pages:
        for e in els:
            if e.cls == "text" and not e.skip:
                c.update(WORD_RE.findall(e.text))
    return c


def fix_title_words(text, vocab, by_len):
    def fix(m):
        w = m.group(0)
        if len(w) < 4 or vocab.get(w, 0) > 0:
            return w
        best, bc = None, 2
        for L in (len(w),):
            for v in by_len.get(L, ()):
                if vocab[v] > bc and edit1(w, v):
                    best, bc = v, vocab[v]
        return best or w
    return WORD_RE.sub(fix, text)


def correct_titles(arts, pages):
    vocab = build_vocab(pages)
    by_len = {}
    for w, n in vocab.items():
        if n >= 2:
            by_len.setdefault(len(w), []).append(w)
    for a in arts:
        for e in [a.head] + a.kickers + a.decks:
            if not getattr(e, "fixed", False):
                e.text = fix_title_words(e.text, vocab, by_len)
                e.fixed = True


AD_RE = re.compile(r"(নিলাম|টেন্ডার|দরপত্র|e-?Tender|e-?GP|www\.|https?://|\bPLC\b|\bLtd\b|লিমিটেড|পিএলসি|শুভ জন্মদিন|"
                   r"আমরা গভীরভাবে|সংবাদ বিজ্ঞপ্তি|পলাতক আসামি|নিয়োগ বিজ্ঞপ্তি|Office of|Tel:|\bNotice\b|\bJOB\b)", re.I)


def ad_score(a, H):
    text = head_text(a) + " " + " ".join(t for _, t in getattr(a, "body", []))
    hits = len(set(m.lower() for m in AD_RE.findall(text)))
    sc = 2 if hits >= 2 else hits
    has_byline = bool(a.bylines or a.tags or a.jump_out)
    yc = (a.box[1] + a.box[3]) / 2 / H
    if yc > 0.62 and not has_byline:
        sc += 2
    if bn_ratio(head_text(a)) < 0.6:
        sc += 2
    if re.search(r"বিজ্ঞপ্তি|নিলাম|টেন্ডার|e-?Tender|শুভ জন্মদিন|শোকাহত|ক্রয়ের", head_text(a), re.I):
        sc += 2
    last = a.body[-1][1] if getattr(a, "body", None) else ""
    if not has_byline and len(last) < 60 and re.search(r"লিমিটেড|পিএলসি|\bPLC\b|\bLtd\b", last):
        sc += 2          # company signature line closing a paid notice
    return sc


COLORS = [(0, 0, 255), (0, 140, 0), (255, 0, 0), (0, 160, 255), (200, 0, 200), (160, 160, 0), (0, 200, 200), (100, 0, 160)]


def write_debug(pg, img, arts, els, path):
    out = img.copy()
    for e in els:
        if e.skip and e.skip not in ("merged",):
            cv2.rectangle(out, (int(e.x0), int(e.y0)), (int(e.x1), int(e.y1)), (150, 150, 150), 4)
            cv2.putText(out, e.skip[:8], (int(e.x0) + 4, int(e.y0) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
    for k, a in enumerate(arts):
        col = COLORS[k % len(COLORS)]
        parts = [a.head] + a.kickers + a.decks + a.bylines + a.tags + a.blocks
        for e in parts:
            th = 10 if e is a.head else 5
            cv2.rectangle(out, (int(e.x0), int(e.y0)), (int(e.x1), int(e.y1)), col, th)
        for f, c in a.photos:
            cv2.rectangle(out, (int(f.x0), int(f.y0)), (int(f.x1), int(f.y1)), col, 12)
            if c is not None:
                cv2.rectangle(out, (int(c.x0), int(c.y0)), (int(c.x1), int(c.y1)), col, 5)
        cv2.putText(out, a.id or str(k), (int(a.head.x0) + 8, int(a.head.y0) - 8), cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, 6)
    cv2.imwrite(str(path), cv2.resize(out, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA))


def run_all(pdf, out_dir):
    out_dir = Path(out_dir)
    pages = process(pdf, out_dir)
    page_arts = {}
    arts = []
    for pg, img, els in pages:
        clusters, lefts, cw = build_page(pg, img, els)
        H, W = img.shape[:2]
        sec = page_section(pg, W)
        base = 18 if sec == "Naksha" else 1
        for k, a in enumerate(sorted(clusters, key=lambda c: (c.head.y0, c.head.x0))):
            a.id = f"p{pg:02d}-{k + 1:02d}"
            a.section = sec
            a.label = str(pg - base + 1)
            a.body = finalize_body(a, cw)
            a.pg_img = img
            arts.append(a)
        page_arts[pg] = clusters
    correct_titles(arts, pages)
    Hs = {pg: img.shape[0] for pg, img, _ in pages}
    for a in arts:
        a.ad_score = ad_score(a, Hs[a.page])
        a.ad = a.ad_score >= 4 and not a.jump_out and not a.tags
    link_jumps(arts, None)
    return pages, arts, page_arts


def slugify(t, n=6):
    words = re.findall(r"[ঀ-৿A-Za-z0-9]+", t)[:n]
    return "-".join(words)[:60] or "untitled"


def art_record(a, out_dir, img_dir):
    seq = chain(a)
    pdf_pages = sorted({x.page for x in seq})
    body, images, notes = [], [], []
    for part in seq:
        body += [dict(type=t, text=x) for t, x in part.body]
        for k, (f, cap) in enumerate(part.photos):
            name = f"{part.id}_{k + 1}.jpg"
            x0, y0, x1, y1 = [int(v) for v in f.box]
            crop = part.pg_img[max(0, y0):y1, max(0, x0):x1]
            cv2.imwrite(str(img_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            images.append(dict(file="images/" + name, caption=clean(cap.text.replace(chr(10), " ")) if cap else "",
                               pdf_page=part.page, box=[x0, y0, x1, y1]))
    kicker = clean(" | ".join(k.text.replace(chr(10), " ") for k in a.kickers))
    deck = clean(" ".join(d.text.replace(chr(10), " ") for d in a.decks))
    byline = clean(" | ".join([b.text.replace(chr(10), " ") for b in a.bylines] + ([a.byline_extra] if a.byline_extra else [])))
    flags = []
    if a.jump_out and getattr(a, "cont", None) is None:
        flags.append(f"jump to page {a.jump_out[0]} col {a.jump_out[1]} not linked")
    if a.tags and not any(x.cont is a for x in ALL_ARTS):
        flags.append("continuation not linked to a source")
    if not body:
        flags.append("no body text")
    return dict(id=a.id, section=a.section, pdf_pages=pdf_pages, kicker=kicker, headline=head_text(a), deck=deck,
                byline=byline, body=body, text=(chr(10) * 2).join(b["text"] for b in body), images=images, flags=flags,
                segments=[dict(pdf_page=p.page, region=[round(v) for v in p.box]) for p in seq])


def write_outputs(out_dir, arts, source):
    global ALL_ARTS
    ALL_ARTS = arts
    out_dir = Path(out_dir)
    (out_dir / "articles").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "articles").glob("*.md"):
        old.unlink()
    img_dir = out_dir / "images"; img_dir.mkdir(exist_ok=True)
    continuations = {id(getattr(a, "cont", None)) for a in arts if getattr(a, "cont", None)}
    recs, dropped = [], []
    for a in sorted(arts, key=lambda x: x.id):
        if id(a) in continuations:
            continue
        if a.ad or (not a.body and not a.photos):
            dropped.append(dict(id=a.id, page=a.page, headline=head_text(a), reason="ad" if a.ad else "no body",
                                ad_score=a.ad_score))
            continue
        recs.append(art_record(a, out_dir, img_dir))
    for r in recs:
        md = [f"# {r['headline']}"]
        if r["kicker"]:
            md.append(f"**{r['kicker']}**")
        if r["deck"]:
            md.append(f"_{r['deck']}_")
        meta = " — ".join(x for x in [r["byline"], r["section"], "PDF p" + ", p".join(str(p) for p in r["pdf_pages"])] if x)
        md.append(f"_{meta}_")
        for im in r["images"]:
            md.append(f"![{im['caption']}]({'../' + im['file']})" + (chr(10) * 2 + f"_{im['caption']}_" if im["caption"] else ""))
        for b in r["body"]:
            md.append(("## " if b["type"] == "h" else "") + b["text"])
        (out_dir / "articles" / f"{r['id']}-{slugify(r['headline'])}.md").write_text((chr(10) * 2).join(md) + chr(10), encoding="utf8")
    (out_dir / "articles.json").write_text(
        json.dumps(dict(source=source, articles=recs, excluded=dropped), ensure_ascii=False, indent=1), encoding="utf8")
    # html index
    h = ['<!doctype html><html lang="bn"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>Prothom Alo Articles</title><style>body{font:17px/1.7 'Noto Sans Bengali','Nirmala UI',sans-serif;max-width:780px;margin:0 auto;padding:0 16px;background:#fbfaf7;color:#1d1d1b}"
         "article{border-top:1px solid #ddd;padding:24px 0}h2{margin:4px 0}.m{color:#777;font-size:13px}.k{color:#a33;font-size:13px}"
         ".flag{color:#b00020;font-size:12px}img{max-width:100%}figcaption{font-size:13px;color:#666}</style></head><body>",
         f"<h1>Prothom Alo — extracted articles</h1><div class=m>Source: {html.escape(source)} · {len(recs)} articles</div><ol>"]
    for r in recs:
        h.append(f"<li><a href='#{r['id']}'>{html.escape(r['headline'])}</a> <span class=m>({r['section']} p{r['pdf_pages'][0]})</span></li>")
    h.append("</ol>")
    for r in recs:
        h.append(f"<article id='{r['id']}'>")
        if r["kicker"]:
            h.append(f"<div class=k>{html.escape(r['kicker'])}</div>")
        h.append(f"<h2>{html.escape(r['headline'])}</h2>")
        if r["deck"]:
            h.append(f"<div class=m>{html.escape(r['deck'])}</div>")
        h.append(f"<div class=m>{html.escape(r['byline'])} · {r['section']} · PDF p{', p'.join(str(p) for p in r['pdf_pages'])}</div>")
        for f in r["flags"]:
            h.append(f"<div class=flag>⚠ {html.escape(f)}</div>")
        for im in r["images"]:
            h.append(f"<figure><img src='{im['file']}'><figcaption>{html.escape(im['caption'])}</figcaption></figure>")
        for b in r["body"]:
            h.append(f"<h4>{html.escape(b['text'])}</h4>" if b["type"] == "h" else f"<p>{html.escape(b['text'])}</p>")
        h.append("</article>")
    h.append("</body></html>")
    (out_dir / "index.html").write_text(chr(10).join(h), encoding="utf8")
    return recs


def main():
    pdf = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(Path(pdf).stem.replace(" Daily_ePapers_BD", "") + " articles")
    pages, arts, page_arts = run_all(pdf, out)
    dbg = out / "debug"; dbg.mkdir(parents=True, exist_ok=True)
    for pg, img, els in pages:
        write_debug(pg, img, page_arts[pg], els, dbg / f"p{pg:02d}.png")
    recs = write_outputs(out, arts, Path(pdf).name)
    print("articles", len(arts), "written", len(recs))
    for a in arts:
        print(a.id, "|", head_text(a)[:50], "|", len(a.body), "paras", "| jump", a.jump_out, "| tag" if a.tags else "", "| ->", getattr(a, "cont", None) and a.cont.id)


if __name__ == "__main__":
    main()

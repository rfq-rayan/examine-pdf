#!/usr/bin/env python3
"""Rule-based article extractor for The Daily Star e-paper PDF.

The Daily Star PDF is the publisher's own vector output (per-page PDFs run
through Ghostscript pdfwrite, then merged with PyPDF2), so everything here
works from the real text layer -- no OCR:

  1. Rebuild visual text lines (MuPDF splits widely-justified lines).
  2. Give every line a role from its font/size (body, headline, byline, ...).
  3. Drop text hidden under a later-drawn image (leftover layout frames).
  4. Group body lines into "story runs": the content stream emits a story's
     threaded text frames consecutively; breaks come from bylines,
     "FROM PAGE" lines, headlines and column-flow discontinuities.
  5. Give each run its headline by geometry (nearest headline above, nothing
     in between) -- headlines are NOT reliably emitted before their body.
  6. Attach photos, captions, credits, pull quotes by position.
  7. Link "SEE PAGE n COL m" jumps to "FROM PAGE n" continuations using the
     column number plus headline similarity, and merge them.

Usage:
    python extract_dailystar.py "Daily Star 06-10-2026 Daily_ePapers_BD.pdf" [out_dir]
"""
import html
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# ----------------------------------------------------------------------------
# Patterns and constants
# ----------------------------------------------------------------------------
# "SEE PAGE 2 COL 4" (main), "READ MORE ON B3" (business), "CONTINUED TO PAGE 11" (Lifestyle)
JUMP_OUT_RE = re.compile(r"^(?:SEE\s+PAGE\s+([A-Z]?\d+)\s+COL\s+(\d+)|READ\s+MORE\s+ON\s+(?:PAGE\s+)?([A-Z]?\d+)"
                         r"|CONTINUED\s+(?:TO|ON)\s+PAGE\s+([A-Z]?\d+))\s*$", re.I)
JUMP_IN_RE = re.compile(r"^(?:CONTINUED\s+)?FROM\s+PAGE\s+([A-Z]?\d+)\b\s*(.*)$", re.I)
CREDIT_RE = re.compile(r"^(PHOTO|PHOTOS|FILE PHOTO|ILLUSTRATION|GRAPHIC|GRAPHICS|SOURCE|COURTESY|CARTOON)\b", re.I)
DATE_RE = re.compile(r"(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\s+\w+\s+\d+,\s+\d{4}|\b\d{4}\s+BS\b", re.I)
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", " ": " ", "­": ""}
TERMINAL = tuple('.!?:"”’)')
STOPWORDS = set("a an the of to in on for and or with as at by from is are was were be its it this that".split())

# Front-page masthead band (ears, teasers, edition info) and running-header band.
FRONT_MASTHEAD_Y = 245
HEADER_BAND_Y = 95
# Main-paper 6-column grid (left edges, pt). Continuation pages shift the
# right half slightly, so the nearest anchor is used.
MAIN_COL_X = [36, 192, 348, 515, 671, 826]


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass(eq=False)
class Line:
    page: int
    idx: int  # content-stream order on the page
    block: int
    rect: pymupdf.Rect
    text: str
    font: str
    size: float
    bold: bool
    italic: bool
    plain: bool = True  # has at least one span that is neither bold nor italic
    role: str = ""
    hidden: bool = False

    @property
    def words(self):
        return len(self.text.split())


@dataclass(eq=False)
class Img:
    page: int
    idx: int
    rect: pymupdf.Rect
    xref: int
    width: int
    height: int
    has_mask: bool = False
    raw: bytes = b""
    ext: str = ""
    caption: str = ""
    credit: str = ""
    cap_rect: pymupdf.Rect = None
    owner: "Run" = None
    file: str = ""


@dataclass(eq=False)
class HeadGroup:
    page: int
    lines: list
    rect: pymupdf.Rect
    kicker: str = ""
    headline: str = ""
    deck: str = ""
    is_label: bool = False
    owner: "Run" = None


@dataclass(eq=False)
class Run:
    page: int
    lines: list = field(default_factory=list)  # BODY/SUBHEAD/BYLINE/JUMP_IN lines in stream order
    head: HeadGroup = None
    labels: list = field(default_factory=list)
    jump_out: list = field(default_factory=list)  # (page_label, col or None)
    images: list = field(default_factory=list)
    pull_quotes: list = field(default_factory=list)
    notes: list = field(default_factory=list)  # author bios, attributions, stray captions
    extras: list = field(default_factory=list)  # graphic/table text
    flags: list = field(default_factory=list)

    @property
    def body_lines(self):
        return [l for l in self.lines if l.role in ("BODY", "SUBHEAD")]

    @property
    def rect(self):
        r = pymupdf.Rect(self.lines[0].rect)
        for l in self.lines[1:]:
            r |= l.rect
        return r

    def region(self):
        r = self.rect
        if self.head:
            r |= self.head.rect
        return r

    @property
    def jump_in(self):
        for l in self.lines:
            if l.role == "JUMP_IN":
                return norm_label(JUMP_IN_RE.match(l.text).group(1))
        return None


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def clean(text):
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def font_parts(font):
    font = font.split("+")[-1]
    fam, _, style = font.partition("-")
    return fam, style or "Regular"


def is_boldish(style):
    return any(w in style for w in ("Bold", "Black", "Heavy", "CnBold"))


def is_caps(text):
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def h_overlap(a, b):
    return min(a.x1, b.x1) - max(a.x0, b.x0)


def tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def slugify(text, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "untitled"


def norm_label(s):
    """'02' -> '2', 'b3' -> 'B3'."""
    m = re.fullmatch(r"([A-Za-z]?)0*(\d+)", s.strip())
    return (m.group(1).upper() + m.group(2)) if m else s.strip()


def is_letterspaced(text):
    toks = text.split()
    return len(toks) >= 4 and sum(len(t) == 1 for t in toks) >= 0.6 * len(toks)


def collapse_letterspacing(chars):
    """'I N F O C U S' -> 'IN FOCUS' using the real gaps between glyphs."""
    glyphs = [c for c in chars if c["c"].strip()]
    if len(glyphs) < 4:
        return None
    gaps = [b["bbox"][0] - a["bbox"][2] for a, b in zip(glyphs, glyphs[1:])]
    med = statistics.median(gaps)
    out = glyphs[0]["c"]
    for g, ch in zip(gaps, glyphs[1:]):
        out += (" " if g > med * 1.6 + 0.5 else "") + ch["c"]
    return out


# ----------------------------------------------------------------------------
# 1. Visual lines and images, in content-stream order
# ----------------------------------------------------------------------------
def read_page(page, pno):
    raw = page.get_text("rawdict", sort=False)
    img_info = {i["number"]: i for i in page.get_image_info(xrefs=True)}
    lines, images, idx = [], [], 0
    for b in raw["blocks"]:
        if b["type"] == 1:
            info = img_info.get(b["number"], {})
            xref = info.get("xref", 0)
            has_mask = bool(xref and page.parent.extract_image(xref).get("smask"))
            images.append(Img(pno, idx, pymupdf.Rect(b["bbox"]), xref, b["width"], b["height"], has_mask,
                              b.get("image", b"") if not xref else b"", b.get("ext", "png")))
            idx += 1
            continue
        # Merge MuPDF lines that share a baseline (wide justification splits them).
        visual = []
        for l in b["lines"]:
            if not l["spans"] or l["dir"][0] < 0.99:
                continue
            base = l["spans"][0]["origin"][1]
            size = l["spans"][0]["size"]
            if visual and abs(visual[-1]["base"] - base) < 1.2 and abs(visual[-1]["size"] - size) < 0.6                     and -2 <= l["bbox"][0] - visual[-1]["rect"].x1 <= 2.5 * size:
                visual[-1]["spans"].extend(l["spans"])
                visual[-1]["rect"] |= l["bbox"]
            else:
                visual.append({"base": base, "size": size, "spans": list(l["spans"]), "rect": pymupdf.Rect(l["bbox"])})
        for v in visual:
            parts, weight, prev_x1, styles = [], {}, None, []
            for s in v["spans"]:
                t = "".join(c["c"] for c in s["chars"])
                if prev_x1 is not None and s["chars"] and s["chars"][0]["bbox"][0] - prev_x1 > s["size"] * 0.15 \
                        and parts and not parts[-1].endswith(" ") and not t.startswith(" "):
                    parts.append(" ")
                parts.append(t)
                if s["chars"]:
                    prev_x1 = s["chars"][-1]["bbox"][2]
                key = (s["font"], round(s["size"], 1), s["flags"])
                n = len(t.strip().replace("▌", ""))
                weight[key] = weight.get(key, 0) + n
                if n:
                    styles.append((font_parts(s["font"])[1], s["flags"]))
            text = clean("".join(parts))
            if not text:
                continue
            if is_letterspaced(text):  # 'I N F O C U S', 'A N A LY S I S'
                text = collapse_letterspacing([c for s in v["spans"] for c in s["chars"]]) or text
            font, size, flags = max(weight, key=weight.get)
            fam, style = font_parts(font)
            all_bold = bool(styles) and all(is_boldish(st) or fl & 16 for st, fl in styles)
            all_italic = bool(styles) and all("Italic" in st or st.endswith("It") or fl & 2 for st, fl in styles)
            plain = any(not (is_boldish(st) or fl & 16) and not ("Italic" in st or st.endswith("It") or fl & 2)
                        for st, fl in styles)
            lines.append(Line(pno, idx, b["number"], v["rect"], text, f"{fam}-{style}", size, all_bold, all_italic,
                              plain))
            idx += 1
    return lines, images


# ----------------------------------------------------------------------------
# 2. Roles
# ----------------------------------------------------------------------------
def role_main(l, pno, page_h):
    t = l.text
    fam, style = font_parts(l.font)
    if JUMP_OUT_RE.match(t):
        return "JUMP_OUT"
    if JUMP_IN_RE.match(t):
        return "JUMP_IN"
    if "▌" in t:  # the ▌ bar that precedes every byline
        return "BYLINE"
    if pno == 0 and l.rect.y1 < FRONT_MASTHEAD_Y:
        return "FURNITURE"
    if l.rect.y1 < HEADER_BAND_Y and (l.size >= 40 or DATE_RE.search(t) or "@" in t
                                      or re.fullmatch(r"[A-Z]?\d{1,2}|city|2nd edition", t, re.I)):
        return "FURNITURE"
    if l.rect.y0 > page_h - 60 and l.size <= 8:
        return "FURNITURE"  # imprint / contact strip at the foot of the page
    if style.startswith("Light") and l.size < 12:
        return "FURNITURE"  # imprint, "FOUNDER EDITOR" strap
    if fam.startswith("Headlines"):
        return "GRAPHIC"  # infographics, tables, TV guide
    if fam != "AntoniaH1":
        return "OTHER"  # advertisement text
    if CREDIT_RE.match(t):
        return "CREDIT"
    if l.size < 7.6:
        return "CAPTION" if l.bold else "OTHER"  # name labels under mugshots
    if 8.3 <= l.size <= 9.7:
        if l.italic or (not l.plain and not l.bold):  # italic, or bold name + italic text
            return "BIO"
        if l.bold:
            return "SUBHEAD" if is_caps(t) else "ATTRIB"
        return "BODY"
    if 9.7 < l.size <= 10.6:
        return "CAPTION" if l.bold else "ATTRIB"
    if 11.4 <= l.size <= 12.6 and l.bold:
        return "LABEL" if is_caps(t) and len(t) < 30 else "PULLQUOTE"
    if l.size >= 13.5:
        return "HEADLINE"
    return "OTHER"


def role_lifestyle(l, pno, page_h):
    t = l.text
    fam, style = font_parts(l.font)
    if JUMP_OUT_RE.match(t):
        return "JUMP_OUT"
    if JUMP_IN_RE.match(t):
        return "JUMP_IN"
    if fam.startswith("CampaignFranklin") and re.fullmatch(r"\d{1,2}", t):
        return "FURNITURE"  # folio
    if re.match(r"^VOLUME \d+", t, re.I) or fam.startswith("Blinker"):
        return "FURNITURE"
    if fam.startswith("Existence"):
        return "PULLQUOTE"  # display face used for standfirsts / pull quotes
    if fam.startswith("Myriad"):
        if "BlackCond" in style:
            return "LABEL"  # '#FASHION & BEAUTY'
        if l.size >= 13.5:
            return "HEADLINE"
        if style == "Bold":
            if t.lower().startswith("by "):
                return "BYLINE"
            return "SUBHEAD" if is_caps(t) and len(t.split()) >= 2 and ":" not in t else "CREDIT"
        if "BoldCond" in style or "CnBold" in style:
            return "SUBHEAD"
        if 7.0 <= l.size <= 9.7:
            return "CREDIT" if is_caps(t) and len(t) < 40 else "BODY"
        return "OTHER"
    if l.size >= 13.5:
        return "HEADLINE"
    return "OTHER"


def fix_inline_styles(lines):
    """A body line that happens to be all italic (a book title) or all bold sits
    tight between two body lines of the same column: it is body text. A one-word
    capitals line right under a crosshead finishes that crosshead."""
    vis = [l for l in lines if not l.hidden]
    for i, l in enumerate(vis):
        prev = vis[i - 1] if i else None
        nxt = vis[i + 1] if i + 1 < len(vis) else None
        if l.role in ("BIO", "ATTRIB") and prev and nxt and prev.role == "BODY" and nxt.role == "BODY":
            above, below = l.rect.y0 - prev.rect.y1, nxt.rect.y0 - l.rect.y1
            aligned = h_overlap(prev.rect, l.rect) > 0.5 * l.rect.width and h_overlap(nxt.rect, l.rect) > 0.5 * l.rect.width
            if aligned and abs(below) < 4:
                if abs(above) < 4:
                    l.role = "BODY"  # inline italic/bold line inside a paragraph
                elif l.role == "ATTRIB" and above < 15:
                    l.role = "SUBHEAD"  # title-case bold crosshead ('Remembering Gandhi')
        elif l.role == "CREDIT" and prev and prev.role == "SUBHEAD" and is_caps(l.text)                 and abs(l.rect.y0 - prev.rect.y1) < 4 and h_overlap(prev.rect, l.rect) > 0:
            l.role = "SUBHEAD"


def mark_hidden(lines, images):
    """Text drawn *before* an opaque image that covers it is invisible."""
    for l in lines:
        area = l.rect.get_area() or 1
        for im in images:
            if im.idx > l.idx and not im.has_mask and (l.rect & im.rect).get_area() / area > 0.8:
                l.hidden = True
                break


# ----------------------------------------------------------------------------
# 3. Headline groups (built first: runs use them as separators)
# ----------------------------------------------------------------------------
def build_headgroups(lines, pno):
    cand = [l for l in lines if l.role in ("HEADLINE", "LABEL") and not l.hidden]
    parent = list(range(len(cand)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(cand):
        for j in range(i + 1, len(cand)):
            b = cand[j]
            if abs(a.idx - b.idx) > 10:
                continue
            vgap = max(b.rect.y0 - a.rect.y1, a.rect.y0 - b.rect.y1)
            if vgap <= max(8, 0.45 * max(a.size, b.size)) and h_overlap(a.rect, b.rect) > 0:
                parent[find(i)] = find(j)
    groups = {}
    for i, l in enumerate(cand):
        groups.setdefault(find(i), []).append(l)

    out = []
    for ls in groups.values():
        ls.sort(key=lambda l: (round(l.rect.y0), l.rect.x0))
        g = HeadGroup(pno, ls, rect_of(ls))
        heads = [l for l in ls if l.role == "HEADLINE"]
        if not heads or (len(heads) == 1 and is_caps(heads[0].text) and heads[0].size <= 20
                         and len(heads[0].text.split()) <= 4):
            g.is_label = True  # 'BAY OF BENGAL CONVERSATION', 'IN FOCUS', 'ANALYSIS'
            g.headline = " ".join(l.text for l in ls)
            out.append(g)
            continue
        # main headline style = the (font, size) covering the most visual area
        score = {}
        for l in heads:
            score[(l.font, l.size)] = score.get((l.font, l.size), 0) + len(l.text) * l.size ** 2
        mfont, msize = max(score, key=score.get)
        main_caps = is_caps(" ".join(l.text for l in heads if (l.font, l.size) == (mfont, msize)))
        first_main = min(l.rect.y0 for l in heads if (l.font, l.size) == (mfont, msize))
        head_lines, kicker, deck = [], [], []
        for l in ls:
            same = l.font == mfont and l.size >= 0.6 * msize
            if l.rect.y1 <= first_main + 2 and (l.role == "LABEL" or not same or l.size < 0.62 * msize
                                                or (is_caps(l.text) and not main_caps)):
                kicker.append(l)
            elif same:
                head_lines.append(l)
            elif l.role == "HEADLINE" and head_lines:
                deck.append(l)
            else:
                kicker.append(l) if not head_lines else deck.append(l)
        g.kicker = " ".join(l.text for l in kicker)
        g.headline = join_text([l.text for l in head_lines])
        g.deck = " ".join(l.text for l in deck if l.role == "HEADLINE")
        out.append(g)
    return out


def join_text(parts):
    out = ""
    for p in parts:
        if not out:
            out = p
        elif out.endswith("-") and p[:1].islower():
            out += p
        else:
            out += " " + p
    return out


def rect_of(ls):
    r = pymupdf.Rect(ls[0].rect)
    for l in ls[1:]:
        r |= l.rect
    return r


# ----------------------------------------------------------------------------
# 4. Story runs
# ----------------------------------------------------------------------------
RUN_ROLES = ("BODY", "SUBHEAD", "BYLINE", "JUMP_IN")


def build_runs(lines, pno, lifestyle, groups):
    head_rects = [g.rect for g in groups if not g.is_label]
    runs, cur, col, hard_sep = [], None, None, False

    def headline_between(a, b):
        return any(h_overlap(h, a) > 5 and h.y0 >= a.y1 - 1 and h.y1 <= b.y0 + 1 for h in head_rects)

    def continues(prev, l):
        if hard_sep:
            return False
        if l.role == "BYLINE" and not lifestyle:
            # a wrapped byline repeats the ▌ bar on each line
            return all(x.role == "BYLINE" for x in cur.lines) and -6 <= l.rect.y0 - prev.rect.y1 < 8
        if l.role == "JUMP_IN":
            return False
        # same column (judged against the whole column, not a short last line), moving down
        if h_overlap(col, l.rect) >= 0.5 * min(col.width, l.rect.width) and l.rect.y0 >= prev.rect.y0 - 2:
            return not headline_between(pymupdf.Rect(col.x0, prev.rect.y0, col.x1, prev.rect.y1), l.rect)
        # next column to the right (gutter <= 30pt)
        return col.x1 - 4 <= l.rect.x0 <= col.x1 + 30

    for l in lines:
        if l.hidden:
            continue
        if l.role in RUN_ROLES:
            if cur is None or not continues(cur.lines[-1], l):
                cur = Run(pno)
                runs.append(cur)
                col = pymupdf.Rect(l.rect)
            elif h_overlap(col, l.rect) > 0:
                col |= l.rect
            else:
                col = pymupdf.Rect(l.rect)
            cur.lines.append(l)
            hard_sep = lifestyle and l.role == "BYLINE"  # Lifestyle bylines close an article
        elif l.role in ("HEADLINE", "FURNITURE"):
            hard_sep = True

    # runs without any body text (a byline frame emitted on its own) join the
    # run that starts right below them (main paper) or ends right above (Lifestyle)
    out = []
    for r in runs:
        if r.body_lines:
            out.append(r)
            continue
        rr = r.rect
        target = None
        for o in runs:
            if o is r or not o.body_lines:
                continue
            if not lifestyle and 0 <= o.rect.y0 - rr.y1 <= 30 and h_overlap(o.lines[0].rect, rr) > 0:
                target, front = o, True
            elif lifestyle and 0 <= rr.y0 - max(l.rect.y1 for l in o.lines) <= 60 and h_overlap(o.rect, rr) > 0:
                target, front = o, False
        if target:
            if front:
                target.lines[0:0] = r.lines
            else:
                target.lines.extend(r.lines)
        else:
            out.append(r)
    return out


def assign_headlines(runs, groups, page_lines, images):
    """Nearest headline above each run's first line, with no body text or other
    headline in between. Photos, captions or graphics may sit in between, which
    allows a larger gap. The closest run owns a headline; leftover runs continue
    the previous run when they plausibly sit next to it."""
    heads = [g for g in groups if not g.is_label]
    run_of = {id(l): r for r in runs for l in r.lines}
    body = [l for r in runs for l in r.lines]
    fillers = [im.rect for im in images] + [l.rect for l in page_lines if not l.hidden and l.role in
                                            ("CAPTION", "CREDIT", "GRAPHIC", "BIO", "ATTRIB", "PULLQUOTE", "LABEL")]

    def blocked(r, band, g):
        if any(run_of[id(l)] is not r and h_overlap(l.rect, band) > 5 and l.rect.y0 >= band.y0 - 1
               and l.rect.y1 <= band.y1 + 1 for l in body):
            return True
        return any(o is not g and h_overlap(o.rect, band) > 5 and o.rect.y0 >= band.y0 - 1
                   and o.rect.y1 <= band.y1 + 1 for o in heads)

    choice = {}
    for r in runs:
        first = r.lines[0]
        col = pymupdf.Rect(first.rect)
        for l in r.lines[1:6]:
            if h_overlap(col, l.rect) > 0:
                col |= l.rect
        best = None
        for g in heads:
            if g.rect.y1 > first.rect.y0 + 0.5 * first.size + 6 or h_overlap(g.rect, col) < min(10, col.width * 0.3):
                continue
            gap = first.rect.y0 - g.rect.y1
            band = pymupdf.Rect(max(col.x0, g.rect.x0), g.rect.y1, min(col.x1, g.rect.x1), first.rect.y0)
            if gap > 150:
                filled = any(h_overlap(f, band) > 5 and f.y0 < band.y1 and f.y1 > band.y0 for f in fillers)
                if gap > 450 or not filled:
                    continue
            if blocked(r, band, g):
                continue
            if best is None or gap < best[0]:
                best = (gap, g)
        if best:
            choice[id(r)] = best
    for r in sorted(runs, key=lambda r: choice.get(id(r), (1e9,))[0]):
        c = choice.get(id(r))
        if c and c[1].owner is None:
            c[1].owner = r
            r.head = c[1]
    # a run pointing at a headline another run owns is the same story (e.g. a
    # wide headline emitted between columns 1-3 and 4-6 of its own body)
    for r in list(runs):
        c = choice.get(id(r))
        if c and c[1].owner is not r and c[1].owner in runs:
            owner = c[1].owner
            owner.lines = sorted(owner.lines + r.lines, key=lambda l: l.idx)
            runs.remove(r)

    # side headlines (headline to the left of the text, common in Lifestyle)
    for r in runs:
        if r.head:
            continue
        first = r.lines[0]
        best = None
        for g in heads:
            if g.owner is not None or g.rect.x1 > first.rect.x0 + 5:
                continue
            dx = first.rect.x0 - g.rect.x1
            if dx > 250 or g.rect.y0 > first.rect.y0 + 60 or g.rect.y1 < first.rect.y0 - 250:
                continue
            band = pymupdf.Rect(g.rect.x1, min(g.rect.y0, first.rect.y0), first.rect.x0, max(g.rect.y1, first.rect.y1))
            if any(run_of[id(l)] is not r and l.rect.x0 >= band.x0 - 1 and l.rect.x1 <= band.x1 + 1
                   and l.rect.y1 > band.y0 and l.rect.y0 < band.y1 for l in body):
                continue
            d = dx + abs(g.rect.y0 - first.rect.y0)
            if best is None or d < best[0]:
                best = (d, g)
        if best:
            best[1].owner = r
            r.head = best[1]

    # headless runs: continuation columns of the previous run, if adjacent
    # (up to one column of photos/icons may separate them, as in grids)
    merged = []
    for r in runs:
        prev = merged[-1] if merged else None
        if r.head is None and prev and r.lines[0].role not in ("JUMP_IN", "BYLINE"):
            pr, fr = prev.rect, r.lines[0].rect
            gap_box = pymupdf.Rect(min(pr.x1, fr.x0), max(pr.y0, fr.y0), max(pr.x1, fr.x0), min(pr.y1, fr.y1))
            between = any(run_of[id(l)] not in (r, prev) and l.rect.intersects(gap_box) for l in body)
            if pr.x0 - 40 <= fr.x0 <= pr.x1 + 120 and pr.y0 - 80 <= fr.y0 <= pr.y1 + 80 and not between:
                prev.lines.extend(r.lines)
                for l in r.lines:
                    run_of[id(l)] = prev
                continue
        merged.append(r)
    return merged


# ----------------------------------------------------------------------------
# 5. Attach everything else by position
# ----------------------------------------------------------------------------
def group_lines(lines, roles):
    """Consecutive (stream) lines of the given roles that touch each other."""
    out = []
    for l in lines:
        if l.hidden or l.role not in roles:
            continue
        if out and out[-1][-1].idx >= l.idx - 2 and abs(l.rect.y0 - out[-1][-1].rect.y1) < l.size * 1.2 \
                and h_overlap(out[-1][-1].rect, l.rect) > -5 and out[-1][-1].role == l.role:
            out[-1].append(l)
        else:
            out.append([l])
    return out


def owner_by_containment(rect, runs, min_frac=0.5):
    best = None
    area = rect.get_area() or 1
    for r in runs:
        reg = r.region()
        frac = (rect & reg).get_area() / area
        if frac >= min_frac and (best is None or reg.get_area() < best[1]):
            best = (r, reg.get_area())
    return best[0] if best else None


def attach(page_lines, images, runs, groups, pno, leftovers):
    # pull-quote attributions that came out as tiny body runs
    pq_rects = [rect_of(g) for g in group_lines(page_lines, ("PULLQUOTE",))]
    keep = []
    for r in runs:
        words = sum(l.words for l in r.lines)
        if r.head is None and len(r.lines) <= 3 and words <= 12 and any(
                0 <= r.rect.y0 - q.y1 <= 25 and h_overlap(q, r.rect) > 0 for q in pq_rects):
            for l in r.lines:
                l.role = "ATTRIB"
            continue
        keep.append(r)
    runs[:] = keep

    # section labels -> articles whose headline sits right under them
    for g in groups:
        if g.is_label:
            for r in runs:
                if r.head and 0 <= r.head.rect.y0 - g.rect.y1 <= 45 and h_overlap(g.rect, r.head.rect) > 0:
                    r.labels.append(g.headline.title())
            inside = owner_by_containment(g.rect, runs, 0.8)
            if inside and g.headline.title() not in inside.labels:
                inside.labels.append(g.headline.title())

    # jump-out lines -> run whose column ends right above them
    for l in page_lines:
        if l.role != "JUMP_OUT" or l.hidden:
            continue
        m = JUMP_OUT_RE.match(l.text)
        if m.group(1):
            target = (norm_label(m.group(1)), int(m.group(2)))
        else:
            target = (norm_label(m.group(3) or m.group(4)), None)
        best = None
        for r in runs:
            for bl in r.lines:
                d = l.rect.y0 - bl.rect.y1
                if -2 <= d <= 25 and h_overlap(bl.rect, l.rect) > 0 and (best is None or d < best[0]):
                    best = (d, r)
        if best is None:  # fall back to the previous run in stream order
            prev = [r for r in runs if r.lines[0].idx < l.idx]
            best = (0, prev[-1]) if prev else None
        if best:
            best[1].jump_out.append(target)
        else:
            leftovers.append(("jump_out", l.text, pno))

    # captions and credits -> images
    caps = group_lines(page_lines, ("CAPTION",))
    for cg in caps:
        cr = rect_of(cg)
        text = join_text([l.text for l in cg])
        best = None
        for im in images:
            below = cr.y0 - im.rect.y1
            side = max(cr.x0 - im.rect.x1, im.rect.x0 - cr.x1)
            if -3 <= below <= 25 and h_overlap(cr, im.rect) >= 0.4 * cr.width:
                d = below
            elif side >= -3 and side <= 20 and min(cr.y1, im.rect.y1) - max(cr.y0, im.rect.y0) > 0:
                d = side + 5
            else:
                continue
            if best is None or d < best[0]:
                best = (d, im)
        if best and not best[1].caption:
            best[1].caption = text
            best[1].cap_rect = cr
        else:
            r = owner_by_containment(cr, runs)
            (r.notes if r else leftovers).append(text if r else ("caption", text, pno))
    for l in page_lines:
        if l.role != "CREDIT" or l.hidden:
            continue
        best = None
        for im in images:
            d = l.rect.y0 - im.rect.y1
            if -15 <= d <= 70 and h_overlap(l.rect, im.rect) > 0 and (best is None or abs(d) < best[0]):
                best = (abs(d), im)
        if best:
            best[1].credit = (best[1].credit + " " + l.text).strip()
        else:
            r = owner_by_containment(l.rect, runs)
            if r:
                r.notes.append(l.text)

    # images -> articles
    for im in images:
        if im.rect.width < 25 or im.rect.height < 25:
            continue
        unit = pymupdf.Rect(im.rect)
        if im.cap_rect:
            unit |= im.cap_rect
        r = owner_by_containment(im.rect, runs, 0.6)
        if r is None:  # article printed on top of the image (front-page cartoon)
            for cand in runs:
                reg = cand.region()
                if (reg & im.rect).get_area() / (reg.get_area() or 1) >= 0.6:
                    r = cand
                    break
        if r is None:  # photo (+caption) directly above its headline, same width as the story
            for cand in runs:
                if cand.head:
                    reg = cand.region()
                    gap = cand.head.rect.y0 - unit.y1
                    ov = h_overlap(unit, reg)
                    if -3 <= gap <= 40 and ov >= 0.8 * unit.width and ov >= 0.8 * reg.width:
                        r = cand
                        break
        if r:
            im.owner = r
            r.images.append(im)

    # pull quotes, bios, attributions, graphic text -> containing article
    for roles, attr in ((("PULLQUOTE",), "pull_quotes"), (("BIO",), "notes"), (("ATTRIB",), "notes"),
                        (("GRAPHIC",), "extras")):
        for g in group_lines(page_lines, roles):
            r = owner_by_containment(rect_of(g), runs, 0.5)
            text = join_text([l.text for l in g])
            if r is None and attr == "notes":
                gr = rect_of(g)
                above = [(gr.y0 - c.rect.y1, c) for c in runs
                         if h_overlap(c.rect, gr) > 0.5 * gr.width and -5 <= gr.y0 - c.rect.y1 <= 40]
                r = min(above, key=lambda t: t[0])[1] if above else None
            if r:
                getattr(r, attr).append(text)
            elif attr != "extras":
                leftovers.append((roles[0].lower(), text, pno))


# ----------------------------------------------------------------------------
# 6. Paragraphs
# ----------------------------------------------------------------------------
def same_column(colr, prev, l):
    return h_overlap(colr, l.rect) > 0.5 * min(colr.width, l.rect.width) and l.rect.y0 >= prev.rect.y0 - 2


def paragraphs(run):
    lines = run.body_lines
    paras, cur, prev, colx0, colr = [], None, None, None, None
    for i, l in enumerate(lines):
        new_col = prev is not None and not same_column(colr, prev, l)
        if prev is None or new_col:
            seg, sr = [l], pymupdf.Rect(l.rect)
            for m in lines[i + 1:]:
                if not same_column(sr, seg[-1], m):
                    break
                seg.append(m)
                sr |= m.rect
            colr, colx0, colw = sr, sr.x0, sr.width
        indent = 3.5 <= l.rect.x0 - colx0 <= 25
        if l.role == "SUBHEAD":
            if paras and paras[-1]["type"] == "subhead" and prev is not None and prev.role == "SUBHEAD":
                paras[-1]["text"] = join_text([paras[-1]["text"], l.text])  # wrapped crosshead
            else:
                paras.append({"type": "subhead", "text": l.text})
            cur, prev = None, l
            continue
        start = cur is None
        if not start:
            ended = prev.text.endswith(TERMINAL)
            short = prev.rect.x1 < colx0 + 0.85 * colw
            if new_col:
                start = indent and (ended or short)
            else:
                start = (indent and (ended or short)) or (l.block != prev.block and ended and short)
        if start:
            cur = {"type": "p", "text": l.text}
            paras.append(cur)
        else:
            cur["text"] = join_text([cur["text"], l.text])
        prev = l
    return paras


# ----------------------------------------------------------------------------
# 7. Page labels, sections, continuations
# ----------------------------------------------------------------------------
def page_labels(doc, lines_by_page):
    """Printed page labels: '1'..'12' main, 'B1'..'B4' business, '1'..'12' Lifestyle."""
    main_w = doc[0].rect.width
    explicit, sections = {}, {}
    for pno, lines in lines_by_page.items():
        for l in lines:
            if l.role != "FURNITURE" or l.rect.y1 > 100:
                continue
            if re.fullmatch(r"[A-Z]?\d{1,2}", l.text):
                explicit[pno] = norm_label(l.text)
            elif l.size >= 40 and re.fullmatch(r"[A-Z &]+(?: \d{1,2})?", l.text):
                sections[pno] = l.text.rstrip("0123456789 ").title()
            elif l.text in ("EDITORIAL", "OPINION"):
                sections.setdefault(pno, l.text.title())
    if 0 not in explicit:
        explicit[0] = "1"

    def shift(label, k):
        m = re.fullmatch(r"([A-Z]?)(\d+)", label)
        return f"{m.group(1)}{int(m.group(2)) + k}"

    def width_units(pno):  # a double-page spread counts as two pages
        return 2 if doc[pno].rect.width > doc[pno].rect.height * 1.2 else 1

    out = dict(explicit)
    for pno in range(doc.page_count):
        if pno in out:
            continue
        same = lambda q: (abs(doc[q].rect.width - main_w) < 1) == (abs(doc[pno].rect.width - main_w) < 1)
        nxt = next((q for q in range(pno + 1, doc.page_count) if q in explicit and same(q)), None)
        prv = next((q for q in range(pno - 1, -1, -1) if q in out and same(q)), None)
        back = shift(explicit[nxt], -sum(width_units(q) for q in range(pno, nxt))) if nxt is not None else None
        fwd = shift(out[prv], sum(width_units(q) for q in range(prv, pno))) if prv is not None else None
        # prefer the next page's numbering when it starts a new prefix (12 -> B1)
        if back and (fwd is None or back[0].isalpha() != fwd[0].isalpha()) and re.search(r"[1-9]", back):
            out[pno] = back
        else:
            out[pno] = fwd or back or str(pno + 1)
    for pno in range(doc.page_count):
        if pno == 0:
            sections[pno] = "Front Page"
        elif abs(doc[pno].rect.width - main_w) >= 1:
            sections[pno] = "Star Lifestyle"
        elif out[pno].startswith("B"):
            sections[pno] = "Star Business"
        else:
            sections.setdefault(pno, sections.get(pno - 1, ""))
    return out, sections


def col_index(x):
    return min(range(len(MAIN_COL_X)), key=lambda i: abs(MAIN_COL_X[i] - x)) + 1


def link_jumps(articles, doc, labels):
    main_w = doc[0].rect.width
    by_label = {}
    for a in articles:
        main = abs(doc[a["_run"].page].rect.width - main_w) < 1
        by_label.setdefault((main, labels[a["_run"].page]), []).append(a)
    used = set()
    for a in articles:
        r = a["_run"]
        main = abs(doc[r.page].rect.width - main_w) < 1
        for tpage, tcol in r.jump_out:
            best = None
            for c in by_label.get((main, tpage), []):
                cr = c["_run"]
                if id(c) in used or cr.jump_in != labels[r.page]:
                    continue
                score = 0.0
                if tcol:
                    ci = col_index(cr.lines[0].rect.x0)
                    score += 2 if ci == tcol else (0.5 if abs(ci - tcol) == 1 else 0)
                ht, ct = tokens(a["headline"]), tokens(c["headline"])
                if ct:
                    score += 3 * len(ht & ct) / len(ct)
                if best is None or score > best[0]:
                    best = (score, c)
            if best and best[0] >= 1.5:
                used.add(id(best[1]))
                a["_continuations"].append(best[1])
                best[1]["_parent"] = a
            else:
                a["flags"].append(f"unmatched jump to page {tpage}" + (f" col {tcol}" if tcol else ""))
    for a in articles:
        if a["_run"].jump_in and "_parent" not in a:
            a["flags"].append(f"continuation from page {a['_run'].jump_in} not matched")


# ----------------------------------------------------------------------------
# 8. Output
# ----------------------------------------------------------------------------
def export_image(doc, im, path):
    try:
        if im.xref:
            pix = pymupdf.Pixmap(doc, im.xref)
            if pix.colorspace is None or pix.colorspace.n not in (1, 3):
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            smask = doc.extract_image(im.xref).get("smask")
            if smask:
                mask = pymupdf.Pixmap(doc, smask)
                if pix.alpha:
                    pix = pymupdf.Pixmap(pix, 0)
                pix = pymupdf.Pixmap(pix, mask)
                path = path.with_suffix(".png")
                pix.save(path)
            else:
                pix.save(path, jpg_quality=92)
            return path
        if im.raw:
            path = path.with_suffix("." + (im.ext or "png"))
            path.write_bytes(im.raw)
            return path
    except Exception:
        pass
    # last resort: render the image area
    pix = doc[im.page].get_pixmap(clip=im.rect, dpi=150)
    pix.save(path)
    return path


def run_to_article(run, labels, sections):
    h = run.head
    byline = dateline = ""
    by_lines = [l.text.replace("▌", "").strip() for l in run.lines if l.role == "BYLINE"]
    if by_lines:
        byline = join_text(by_lines)  # a long byline wraps, with a ▌ bar on every line
        if byline.lower().startswith("by "):
            byline = byline[3:]
        m = re.match(r"^(.*),\s*([A-Z][\w .'-]{1,30})$", byline)
        if m and not m.group(2).isupper():
            byline, dateline = m.group(1).strip(), m.group(2).strip()
    flags = list(run.flags)
    if h is None:
        flags.append("no headline found")
    return {
        "headline": h.headline if h else "",
        "kicker": h.kicker if h else "",
        "deck": h.deck if h else "",
        "labels": run.labels,
        "byline": byline,
        "dateline": dateline,
        "section": sections.get(run.page, ""),
        "flags": flags,
        "_run": run,
        "_continuations": [],
    }


def finalize(a, labels, img_dir, doc, counter):
    parts = [a] + a["_continuations"]
    body, images, pqs, notes, extras, pages, segments = [], [], [], [], [], [], []
    for i, p in enumerate(parts):
        r = p["_run"]
        paras = paragraphs(r)
        if i > 0 and paras and body and body[-1]["type"] == "p" and not body[-1]["text"].endswith(TERMINAL) \
                and paras[0]["type"] == "p":
            body[-1]["text"] = join_text([body[-1]["text"], paras[0]["text"]])  # sentence runs across the jump
            paras = paras[1:]
        body.extend(paras)
        pages.append(r.page + 1)
        segments.append({
            "pdf_page": r.page + 1,
            "page_label": labels[r.page],
            "headline_on_page": p["headline"],
            "region": [round(v, 1) for v in r.region()],
        })
        for im in sorted(r.images, key=lambda im: (im.rect.y0, im.rect.x0)):
            counter[r.page] = counter.get(r.page, 0) + 1
            path = export_image(doc, im, img_dir / f"p{r.page + 1:02d}_{counter[r.page]:02d}.jpg")
            im.file = f"images/{path.name}"
            images.append({"file": im.file, "pdf_page": r.page + 1, "bbox": [round(v, 1) for v in im.rect],
                           "caption": im.caption, "credit": im.credit})
        pqs += r.pull_quotes
        notes += r.notes
        extras += r.extras
        a["flags"] += [f for f in p["flags"] if f not in a["flags"] and i > 0]
    text = "\n\n".join(p["text"] for p in body)
    return {
        "id": "",
        "section": a["section"],
        "pdf_pages": list(dict.fromkeys(pages)),
        "page_labels": list(dict.fromkeys(labels[pg - 1] for pg in pages)),
        "labels": a["labels"],
        "kicker": a["kicker"],
        "headline": a["headline"],
        "deck": a["deck"],
        "byline": a["byline"],
        "dateline": a["dateline"],
        "body": body,
        "text": text,
        "word_count": len(text.split()),
        "images": images,
        "pull_quotes": pqs,
        "notes": notes,
        "graphic_text": extras,
        "segments": segments,
        "flags": a["flags"],
    }


def write_markdown(art, path):
    out = []
    if art["kicker"] or art["labels"]:
        out.append(f"*{' · '.join(art['labels'] + ([art['kicker']] if art['kicker'] else []))}*\n")
    out.append(f"# {art['headline'] or '(no headline)'}\n")
    if art["deck"]:
        out.append(f"**{art['deck']}**\n")
    meta = [x for x in (art["byline"], art["dateline"]) if x]
    pages = ", ".join(f"p{p}" for p in art["pdf_pages"])
    out.append(f"_{' — '.join(meta + [art['section'], 'PDF ' + pages])}_\n")
    for im in art["images"][:1]:
        out.append(f"![{im['caption']}](../{im['file']})\n")
        if im["caption"] or im["credit"]:
            out.append(f"*{im['caption']} {im['credit']}*\n".replace("  ", " "))
    for p in art["body"]:
        out.append(f"### {p['text']}\n" if p["type"] == "subhead" else f"{p['text']}\n")
    for im in art["images"][1:]:
        out.append(f"![{im['caption']}](../{im['file']})\n")
        if im["caption"] or im["credit"]:
            out.append(f"*{im['caption']} {im['credit']}*\n".replace("  ", " "))
    for q in art["pull_quotes"]:
        out.append(f"> {q}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def write_html(arts, photos, out_dir, src_name):
    esc = html.escape
    toc, body = [], []
    for a in arts:
        flag = f' <span class="flag">{esc("; ".join(a["flags"]))}</span>' if a["flags"] else ""
        toc.append(f'<li><a href="#{a["id"]}">{esc(a["headline"] or "(no headline)")}</a> '
                   f'<span class="m">p{",".join(map(str, a["pdf_pages"]))} · {a["word_count"]}w · '
                   f'{len(a["images"])} img</span>{flag}</li>')
        imgs = "".join(f'<figure><img src="{esc(i["file"])}" loading="lazy"><figcaption>{esc(i["caption"])} '
                       f'<i>{esc(i["credit"])}</i></figcaption></figure>' for i in a["images"])
        paras = "".join(f"<h4>{esc(p['text'])}</h4>" if p["type"] == "subhead" else f"<p>{esc(p['text'])}</p>"
                        for p in a["body"])
        pq = "".join(f"<blockquote>{esc(q)}</blockquote>" for q in a["pull_quotes"])
        lab = " · ".join(a["labels"] + ([a["kicker"]] if a["kicker"] else []))
        body.append(f'''<article id="{a["id"]}">
<div class="k">{esc(lab)}</div><h2>{esc(a["headline"] or "(no headline)")}</h2>
{f'<div class="deck">{esc(a["deck"])}</div>' if a["deck"] else ""}
<div class="m">{esc(" — ".join(x for x in (a["byline"], a["dateline"], a["section"]) if x))} · PDF page {", ".join(map(str, a["pdf_pages"]))} · {a["id"]}</div>
{flag}{imgs}{paras}{pq}</article>''')
    ph = "".join(f'<figure><img src="{esc(p["file"])}" loading="lazy"><figcaption>{esc(p["caption"])} '
                 f'<i>{esc(p["credit"])}</i> — PDF p{p["pdf_page"]}</figcaption></figure>' for p in photos)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily Star Articles</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e3e0d8;--accent:#9a3b20}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171715;--fg:#e8e6e0;--muted:#9a978f;--line:#33322e;--accent:#e08a6b}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.6 Georgia,serif;margin:0;padding:0 16px}}
main{{max-width:760px;margin:0 auto}}
h1{{font-size:28px;margin:32px 0 4px}} h2{{font-size:26px;line-height:1.2;margin:4px 0}}
.m{{color:var(--muted);font:13px/1.4 system-ui,sans-serif}} .k{{color:var(--accent);font:600 12px system-ui;text-transform:uppercase;letter-spacing:.06em}}
.deck{{font-size:18px;color:var(--muted);margin:6px 0}} .flag{{color:#b00020;font:12px system-ui}}
article{{border-top:1px solid var(--line);padding:28px 0}} figure{{margin:16px 0}} img{{max-width:100%;height:auto;display:block}}
figcaption{{font:13px/1.4 system-ui;color:var(--muted);margin-top:4px}} blockquote{{border-left:3px solid var(--accent);margin:16px 0;padding-left:12px;font-style:italic}}
h4{{font:700 13px system-ui;text-transform:uppercase;letter-spacing:.05em;margin:20px 0 4px}} ol li{{margin:2px 0}}
</style></head><body><main>
<h1>The Daily Star — extracted articles</h1>
<div class="m">Source: {esc(src_name)} · {len(arts)} articles · {len(photos)} standalone photos</div>
<ol>{"".join(toc)}</ol>
{"".join(body)}
<h2 style="margin-top:40px">Standalone photos</h2>{ph}
</main></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def debug_overlays(doc, runs_by_page, images_by_page, lines_by_page, out_dir):
    colors = [(0.85, 0.1, 0.1), (0.1, 0.45, 0.85), (0.1, 0.6, 0.2), (0.75, 0.4, 0), (0.55, 0.1, 0.7), (0, 0.55, 0.55)]
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)
    for pno in range(doc.page_count):
        page = doc[pno]
        for k, r in enumerate(runs_by_page.get(pno, [])):
            c = colors[k % len(colors)]
            page.draw_rect(r.region(), color=c, width=2.5)
            if r.head:
                page.draw_rect(r.head.rect, color=c, fill=c, fill_opacity=0.12, width=0)
            for im in r.images:
                page.draw_rect(im.rect, color=c, width=1.5, dashes="[6 4] 0")
            page.insert_text(r.region().tl + (3, 14), getattr(r, "aid", "?"), fontsize=13, color=c)
        for im in images_by_page.get(pno, []):
            if im.owner is None:
                page.draw_rect(im.rect, color=(0.5, 0.5, 0.5), width=1, dashes="[2 3] 0")
        for l in lines_by_page.get(pno, []):
            if l.hidden:
                page.draw_rect(l.rect, color=(1, 0, 0), fill=(1, 0, 0), fill_opacity=0.4, width=0)
        page.get_pixmap(dpi=45).save(dbg / f"p{pno + 1:02d}.png")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "Daily Star 06-10-2026 Daily_ePapers_BD.pdf")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem.split(" Daily_ePapers")[0] + " articles")
    for sub in ("images", "articles", "debug"):  # start clean so old runs leave nothing behind
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        for f in (out_dir / sub).iterdir():
            f.unlink()

    doc = pymupdf.open(src)
    main_w = doc[0].rect.width
    lines_by_page, images_by_page, runs_by_page, groups_by_page = {}, {}, {}, {}
    leftovers = []
    for pno in range(doc.page_count):
        page = doc[pno]
        lifestyle = abs(page.rect.width - main_w) >= 1
        lines, images = read_page(page, pno)
        for l in lines:
            l.role = (role_lifestyle if lifestyle else role_main)(l, pno, page.rect.height)
        mark_hidden(lines, images)
        fix_inline_styles(lines)
        groups = build_headgroups(lines, pno)
        runs = build_runs(lines, pno, lifestyle, groups)
        runs = assign_headlines(runs, groups, lines, images)
        attach(lines, images, runs, groups, pno, leftovers)
        lines_by_page[pno], images_by_page[pno], runs_by_page[pno], groups_by_page[pno] = lines, images, runs, groups

    labels, sections = page_labels(doc, lines_by_page)
    articles = [run_to_article(r, labels, sections) for pno in sorted(runs_by_page) for r in runs_by_page[pno]]
    link_jumps(articles, doc, labels)

    # final articles (continuations folded into their parent)
    img_dir = out_dir / "images"
    counter, final, feature_photos = {}, [], []
    for a in articles:
        if "_parent" in a:
            continue
        art = finalize(a, labels, img_dir, doc, counter)
        prose = sum(len(p["text"].split()) for p in art["body"] if p["type"] == "p")
        if prose < 15:  # covers, credit lists, stray labels: no article text
            if art["images"] and art["headline"]:
                feature_photos.append({"file": art["images"][0]["file"], "pdf_page": art["pdf_pages"][0],
                                       "caption": art["headline"], "credit": "; ".join(art["notes"]),
                                       "bbox": art["images"][0]["bbox"]})
            else:
                leftovers.append(("fragment", art["text"], a["_run"].page))
            continue
        art["_run"], art["_src"] = a["_run"], a
        final.append(art)
    final.sort(key=lambda a: (a["pdf_pages"][0], a["_run"].region().y0 // 20, a["_run"].region().x0))
    seen = {}
    for a in final:
        pg = a["pdf_pages"][0]
        seen[pg] = seen.get(pg, 0) + 1
        a["id"] = f"p{pg:02d}-{seen[pg]:02d}"
        a["_run"].aid = a["id"]
        for c in a.get("_continuations", []):
            c["_run"].aid = a["id"] + "c"
    for a in articles:  # continuation runs keep a label for the overlay
        if "_parent" in a:
            a["_run"].aid = getattr(a["_parent"]["_run"], "aid", "?") + "↩"

    # standalone photos (captioned, not inside any article) and text-less features (covers)
    photos = list(feature_photos)
    for pno, ims in images_by_page.items():
        for im in ims:
            if im.owner is None and im.caption:
                counter[pno] = counter.get(pno, 0) + 1
                path = export_image(doc, im, img_dir / f"p{pno + 1:02d}_{counter[pno]:02d}.jpg")
                photos.append({"file": f"images/{path.name}", "pdf_page": pno + 1, "caption": im.caption,
                               "credit": im.credit, "bbox": [round(v, 1) for v in im.rect]})

    for a in final:
        write_markdown(a, out_dir / "articles" / f"{a['id']}-{slugify(a['headline'])}.md")
    clean_arts = [{k: v for k, v in a.items() if not k.startswith("_")} for a in final]

    # self-check: every visible body line should have landed in exactly one article
    placed = set()
    for a in final:
        for part in [a["_src"]] + a["_src"]["_continuations"]:
            placed.update(id(l) for l in part["_run"].lines)
    for pno, ls in lines_by_page.items():
        for l in ls:
            if l.role in ("BODY", "SUBHEAD") and not l.hidden and id(l) not in placed:
                leftovers.append(("unplaced_body", l.text, pno))
    body_words = sum(l.words for ls in lines_by_page.values() for l in ls
                     if l.role in ("BODY", "SUBHEAD") and not l.hidden)
    got_words = sum(len(p["text"].split()) for a in clean_arts for p in a["body"])
    print(f"coverage: {got_words}/{body_words} body words placed in articles "
          f"({100 * got_words / max(body_words, 1):.1f}%)")

    write_html(clean_arts, photos, out_dir, src.name)
    debug_overlays(doc, runs_by_page, images_by_page, lines_by_page, out_dir)
    (out_dir / "articles.json").write_text(json.dumps({
        "source": src.name,
        "articles": clean_arts,
        "standalone_photos": photos,
        "unassigned": [{"kind": k, "text": t, "pdf_page": p + 1} for k, t, p in leftovers],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    flagged = [a for a in clean_arts if a["flags"]]
    print(f"{len(clean_arts)} articles, {len(photos)} standalone photos, "
          f"{sum(len(a['images']) for a in clean_arts)} article images, {len(flagged)} flagged -> {out_dir}")
    for a in flagged:
        print(f"  [{a['id']}] {a['headline'][:60]!r}: {'; '.join(a['flags'])}")


if __name__ == "__main__":
    main()

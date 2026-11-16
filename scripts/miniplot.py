"""A tiny plotting module that writes PDF figures using only the Python standard library.

What this does, in plain words
------------------------------
The project's package list (pyproject.toml / uv.lock) has no plotting library, and adding one
means re-locking the packages.  So this file draws simple charts (lines, dots, axes, a legend)
straight into a PDF.  The report (LaTeX) then includes those PDFs as pictures.

You only need three things from it:
    page = Page(width, height)                         # sizes are in points (72 points = 1 inch)
    ax = Axes(page, left, bottom, width, height, xlim=(0, 1), ylim=(0, 1), xlabel="...", ...)
    ax.line(xs, ys, color=BLUE) / ax.scatter(xs, ys, color=BLUE) / ax.legend([...])
    page.save("figure.pdf")
See make_figures.py for how they are used.  Text is plain ASCII plus a few Latin-1 signs
(for example the "2" in R squared); fonts are the built-in PDF fonts Helvetica and Helvetica-Bold.

Look: thin solid grid lines, 1.5 pt data lines, dots with a white ring so overlaps stay readable,
text in dark grey (never in the data colour), colours from a colour-blind-checked palette.
"""
import math
import os
from contextlib import contextmanager

# Colours: one per "thing" in the report, the same in every figure (model = blue, dummy = orange,
# positive control = aqua).  Checked for colour-blind separation with the dataviz validator.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e4e3de", "#a8a7a1"

# Width of each character in Helvetica, in 1/1000 of the font size, for ASCII 32..126.
_W = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
      556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
      1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
      667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
      333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
      556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]


def _pieces(s):
    """Split a text into (text, size factor, raise factor); "R^2" gives a small raised "2"."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "^" and i + 1 < len(s):
            out.append((s[i + 1], 0.65, 0.38))
            i += 2
        else:
            j = s.find("^", i)
            j = len(s) if j == -1 else j
            out.append((s[i:j], 1.0, 0.0))
            i = j
    return out


def text_width(s, size, bold=False):
    """Approximate width of a text in points (used to centre and right-align text)."""
    w = sum(sum(_W[ord(c) - 32] if 32 <= ord(c) <= 126 else 556 for c in t) * size * f / 1000.0
            for t, f, _ in _pieces(s))
    return w * (1.06 if bold else 1.0)


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return " ".join(f"{int(h[i:i + 2], 16) / 255:.3f}" for i in (0, 2, 4))


def _n(v):
    return f"{v:.2f}"


def nice_ticks(lo, hi, target=5):
    """Round-number tick positions that cover [lo, hi]."""
    span = hi - lo
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    ticks, v = [], first
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def fmt(v):
    return f"{v:g}"


class Page:
    """One PDF page = one figure.  Everything is drawn in points from the bottom-left corner."""

    def __init__(self, width, height):
        self.width, self.height, self.ops = width, height, []

    def line(self, points, color, width=1.5, dash=None):
        d = f"[{dash[0]} {dash[1]}] 0 d" if dash else "[] 0 d"
        path = " ".join(f"{_n(x)} {_n(y)} {'m' if i == 0 else 'l'}" for i, (x, y) in enumerate(points))
        self.ops.append(f"q {_rgb(color)} RG {width} w {d} 1 J 1 j {path} S Q")

    def rect(self, x, y, w, h, fill):
        self.ops.append(f"q {_rgb(fill)} rg {_n(x)} {_n(y)} {_n(w)} {_n(h)} re f Q")

    def circle(self, cx, cy, r, fill, ring="#ffffff", ring_width=0.7):
        k = 0.5523 * r
        path = (f"{_n(cx + r)} {_n(cy)} m "
                f"{_n(cx + r)} {_n(cy + k)} {_n(cx + k)} {_n(cy + r)} {_n(cx)} {_n(cy + r)} c "
                f"{_n(cx - k)} {_n(cy + r)} {_n(cx - r)} {_n(cy + k)} {_n(cx - r)} {_n(cy)} c "
                f"{_n(cx - r)} {_n(cy - k)} {_n(cx - k)} {_n(cy - r)} {_n(cx)} {_n(cy - r)} c "
                f"{_n(cx + k)} {_n(cy - r)} {_n(cx + r)} {_n(cy - k)} {_n(cx + r)} {_n(cy)} c h")
        self.ops.append(f"q {_rgb(fill)} rg {_rgb(ring)} RG {ring_width} w {path} B Q")

    def text(self, x, y, s, size=8, color=INK2, anchor="start", bold=False, rotate=False):
        """Draw text.  Write "R^2" for R with a small raised 2.  rotate=True runs the text upwards."""
        width = text_width(s, size, bold)
        start = {"middle": -width / 2, "end": -width}.get(anchor, 0)
        font = "/F2" if bold else "/F1"
        pen = start                                       # how far along the text we have written
        for piece, factor, lift in _pieces(s):
            safe = piece.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if rotate:   # (x, y) is the middle of the text
                at = f"0 1 -1 0 {_n(x - lift * size)} {_n(y + pen)} Tm"
            else:
                at = f"1 0 0 1 {_n(x + pen)} {_n(y + lift * size)} Tm"
            self.ops.append(f"q {_rgb(color)} rg BT {font} {_n(size * factor)} Tf {at} ({safe}) Tj ET Q")
            pen += text_width(piece, size * factor, bold)

    @contextmanager
    def clipped(self, x, y, w, h):
        """Everything drawn inside this block is cut off outside the rectangle."""
        self.ops.append(f"q {_n(x)} {_n(y)} {_n(w)} {_n(h)} re W n")
        yield
        self.ops.append("Q")

    def save(self, path):
        content = "\n".join(self.ops).encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] /Contents 4 0 R "
             f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>").encode(),
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        ]
        out, offsets = bytearray(b"%PDF-1.4\n"), []
        for i, body in enumerate(objects, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
        out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
        out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        with open(path + ".tmp", "wb") as fh:
            fh.write(bytes(out))
        os.replace(path + ".tmp", path)


class Axes:
    """A chart area inside a Page: axes, grid, tick labels, titles, and the drawing calls."""

    def __init__(self, page, left, bottom, width, height, xlim, ylim, xlabel="", ylabel="", title="",
                 xticks=None, yticks=None):
        self.page, self.l, self.b, self.w, self.h = page, left, bottom, width, height
        self.xlim, self.ylim = xlim, ylim
        xticks = xticks if xticks is not None else nice_ticks(*xlim)
        yticks = yticks if yticks is not None else nice_ticks(*ylim)
        for t in xticks:                                  # grid first, so data sits on top
            page.line([(self.px(t), bottom), (self.px(t), bottom + height)], GRID, 0.5)
        for t in yticks:
            page.line([(left, self.py(t)), (left + width, self.py(t))], GRID, 0.5)
        page.line([(left, bottom), (left + width, bottom)], AXIS, 0.75)
        page.line([(left, bottom), (left, bottom + height)], AXIS, 0.75)
        for t in xticks:
            page.text(self.px(t), bottom - 10, fmt(t), 7.5, INK2, "middle")
        for t in yticks:
            page.text(left - 4, self.py(t) - 2.6, fmt(t), 7.5, INK2, "end")
        if xlabel:
            page.text(left + width / 2, bottom - 22, xlabel, 8, INK2, "middle")
        if ylabel:
            page.text(left - 30, bottom + height / 2, ylabel, 8, INK2, "middle", rotate=True)
        if title:
            page.text(left, bottom + height + 7, title, 8.5, INK, "start", bold=True)

    def px(self, x):
        return self.l + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def py(self, y):
        return self.b + (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def line(self, xs, ys, color, width=1.5, dash=None):
        with self.page.clipped(self.l, self.b, self.w, self.h):
            self.page.line([(self.px(x), self.py(y)) for x, y in zip(xs, ys)], color, width, dash)

    def vline(self, x, color=AXIS, width=0.75):
        self.line([x, x], list(self.ylim), color, width)

    def scatter(self, xs, ys, color, radius=2.6):
        with self.page.clipped(self.l - 4, self.b - 4, self.w + 8, self.h + 8):
            for x, y in zip(xs, ys):
                self.page.circle(self.px(x), self.py(y), radius, color)

    def label(self, x, y, s, anchor="start"):
        """Text placed at data position (x, y), for labelling a line directly."""
        self.page.text(self.px(x), self.py(y), s, 7.5, INK2, anchor)

    def legend(self, entries, corner="lower right", size=7.5):
        """entries: [(text, colour, kind)] with kind "line", "dash" or "dot".  Placed inside the axes."""
        row, pad, sample = 11.5, 5, 18
        width = pad * 2 + sample + 5 + max(text_width(t, size) for t, _, _ in entries)
        height = pad * 2 + row * len(entries) - 3
        x0 = self.l + self.w - width - 4 if "right" in corner else self.l + 4
        y0 = self.b + 4 if "lower" in corner else self.b + self.h - height - 4
        self.page.rect(x0, y0, width, height, "#ffffff")
        for i, (text, color, kind) in enumerate(entries):
            y = y0 + height - pad - 6 - i * row + 1
            if kind == "dot":
                self.page.circle(x0 + pad + sample / 2, y + 2.6, 2.6, color)
            else:
                self.page.line([(x0 + pad, y + 2.6), (x0 + pad + sample, y + 2.6)], color, 1.5, (4, 2.5) if kind == "dash" else None)
            self.page.text(x0 + pad + sample + 5, y, text, size, INK2)

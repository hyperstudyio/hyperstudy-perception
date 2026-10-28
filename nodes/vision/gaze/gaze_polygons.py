"""An object's outline as a gaze candidate region, in numpy alone: OpenCV 5.0.0's cv2.fillPoly and cv2.pointPolygonTest
as the bake-off's gaze_stage.py called them, ported line for line so the regions (and so the targets) are the same.

  fill_polygon(grid, points): cv2.fillPoly(grid, [points], 1.0) with its defaults (8-connected lines, shift 0): every
      edge drawn with an 8-connected Bresenham line (LineIterator, clipped to the grid by clipLine), then the inside
      filled scanline by scanline (FillEdgeCollection). modules/imgproc/src/drawing.cpp (CollectPolyEdges,
      FillEdgeCollection, clipLine, LineIterator::init) and include/opencv2/imgproc.hpp (LineIterator::operator++),
      tag 5.0.0, the OpenCV of opencv-python 5.0.0.93 that both the bake-off and envs/gaze run.
  point_in_polygon(points, x, y): cv2.pointPolygonTest(points as float32, (x, y), False) >= 0: inside or on an edge,
      in float32 coordinates and float64 cross products as OpenCV's float branch computes them
      (modules/geometry/src/geometry.cpp, tag 5.0.0).

Integers here are Python's, which neither overflow nor differ from OpenCV's int64 at these sizes; `>>` floors as
OpenCV's arithmetic shift does, and int64 division, which truncates toward zero, is spelled out (_div).
test_gaze_polygons.py checks both against OpenCV itself where it is installed (envs/gaze), and against rasters
OpenCV made where it is not (CI).
"""

from __future__ import annotations

import numpy as np

XY_SHIFT = 16
XY_ONE = 1 << XY_SHIFT
INT_MAX = 2**31 - 1


def _div(a: int, b: int) -> int:
    """C's integer division: truncates toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def clip_line(width: int, height: int, p1, p2):
    """clipLine(Size, Point&, Point&): (inside, p1, p2), the line clipped to the image (Cohen-Sutherland, with
    OpenCV's double arithmetic truncated to int64). The points move even when the line misses the image."""
    right, bottom = width - 1, height - 1
    (x1, y1), (x2, y2) = p1, p2
    if width <= 0 or height <= 0:
        return False, (x1, y1), (x2, y2)
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if c1 & 12:
            a = 0 if c1 < 8 else bottom
            x1 += int(float(a - y1) * (x2 - x1) / (y2 - y1))
            y1 = a
            c1 = (x1 < 0) + (x1 > right) * 2
        if c2 & 12:
            a = 0 if c2 < 8 else bottom
            x2 += int(float(a - y2) * (x2 - x1) / (y2 - y1))
            y2 = a
            c2 = (x2 < 0) + (x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                a = 0 if c1 == 1 else right
                y1 += int(float(a - x1) * (y2 - y1) / (x2 - x1))
                x1 = a
                c1 = 0
            if c2:
                a = 0 if c2 == 1 else right
                y2 += int(float(a - x2) * (y2 - y1) / (x2 - x1))
                x2 = a
                c2 = 0
    return (c1 | c2) == 0, (x1, y1), (x2, y2)


def _outside(width: int, height: int, p) -> bool:
    """OpenCV's (unsigned)x >= (unsigned)width test: negative counts as outside too."""
    return not (0 <= p[0] < width and 0 <= p[1] < height)


def line_points(width: int, height: int, p1, p2):
    """The pixels Line(img, p1, p2, color, 8) sets: LineIterator(img, p1, p2, 8, leftToRight=true)."""
    if _outside(width, height, p1) or _outside(width, height, p2):
        inside, p1, p2 = clip_line(width, height, p1, p2)
        if not inside:
            return []
    (x1, y1), (x2, y2) = p1, p2
    delta_x = delta_y = 1
    dx, dy = x2 - x1, y2 - y1
    if dx < 0:   # leftToRight
        dx, dy = -dx, -dy
        x1, y1 = x2, y2
    if dy < 0:
        dy, delta_y = -dy, -1
    vert = dy > dx
    if vert:
        dx, dy = dy, dx
        delta_x, delta_y = delta_y, delta_x
    err, plus_delta, minus_delta = dx - (dy + dy), dx + dx, -(dy + dy)
    minus_shift, plus_shift, minus_step, plus_step = delta_x, 0, 0, delta_y   # shift: along x; step: along y
    if vert:
        plus_step, plus_shift = plus_shift, plus_step
        minus_step, minus_shift = minus_shift, minus_step
    out, x, y = [], x1, y1
    for _ in range(dx + 1):
        out.append((x, y))
        plus = err < 0
        err += minus_delta + (plus_delta if plus else 0)
        x += minus_shift + (plus_shift if plus else 0)
        y += minus_step + (plus_step if plus else 0)
    return out


class _Edge:
    __slots__ = ("y0", "y1", "x", "dx", "next")

    def __init__(self, y0=0, y1=0, x=0, dx=0):
        self.y0, self.y1, self.x, self.dx, self.next = y0, y1, x, dx, None


def _collect_edges(grid, points, edges):
    """CollectPolyEdges with line_type 8, shift 0, offset (0, 0): draws each edge and keeps the non-horizontal ones in
    fixed point (x << XY_SHIFT), their ends clipped to the grid."""
    height, width = grid.shape
    pt0 = points[-1]
    pt0 = (int(pt0[0]) << XY_SHIFT, int(pt0[1]))
    for v in points:
        pt1 = (int(v[0]) << XY_SHIFT, int(v[1]))
        pt0c, pt1c = list(pt0), list(pt1)
        t0 = ((pt0[0] + (XY_ONE >> 1)) >> XY_SHIFT, pt0[1])
        t1 = ((pt1[0] + (XY_ONE >> 1)) >> XY_SHIFT, pt1[1])
        for x, y in line_points(width, height, t0, t1):
            grid[y, x] = 1.0
        if _outside(width, height, t0) or _outside(width, height, t1):
            _, t0, t1 = clip_line(width, height, t0, t1)
            if t0[1] != t1[1]:
                pt0c[1], pt1c[1] = t0[1], t1[1]
        pt0c[0], pt1c[0] = t0[0] << XY_SHIFT, t1[0] << XY_SHIFT
        if pt0[1] != pt1[1]:
            dx = _div(pt1c[0] - pt0c[0], pt1c[1] - pt0c[1])
            if pt0[1] < pt1[1]:
                edges.append(_Edge(pt0[1], pt1[1], pt0c[0] + (pt0[1] - pt0c[1]) * dx, dx))
            else:
                edges.append(_Edge(pt1[1], pt0[1], pt1c[0] + (pt1[1] - pt1c[1]) * dx, dx))
        pt0 = pt1


def _fill_edges(grid, edges):
    """FillEdgeCollection: the active-edge scanline fill, spans between pairs of edges, both ends rounded inward."""
    height, width = grid.shape
    total = len(edges)
    if total < 2:
        return
    y_min, y_max = INT_MAX, -INT_MAX - 1
    x_min, x_max = 2**63 - 1, -1   # OpenCV's x_max starts at 0xFFFFFFFFFFFFFFFF, which as an int64 is -1
    for e in edges:
        assert e.y0 < e.y1
        x1 = e.x + (e.y1 - e.y0) * e.dx
        y_min, y_max = min(y_min, e.y0), max(y_max, e.y1)
        x_min, x_max = min(x_min, e.x, x1), max(x_max, e.x, x1)
    if y_max < 0 or y_min >= height or x_max < 0 or x_min >= (width << XY_SHIFT):
        return
    # CmpEdges; edges equal in all three run together until one ends, so their order among themselves draws nothing else
    edges = sorted(edges, key=lambda e: (e.y0, e.x, e.dx))
    edges.append(_Edge(INT_MAX))   # the sentinel
    head = _Edge(INT_MAX)
    i, e = 0, edges[0]
    y_max = min(y_max, height)
    delta = XY_ONE - 1
    for y in range(e.y0, y_max):
        draw = False
        clipline = y < 0
        prelast, last = head, head.next
        while last is not None or e.y0 == y:
            if last is not None and last.y1 == y:   # the edge ends here: out of the active list
                prelast.next = last.next
                last = last.next
                continue
            keep_prelast = prelast
            if last is not None and (e.y0 > y or last.x < e.x):
                prelast, last = last, last.next
            elif i < total:                          # the edge starts here: into the active list
                prelast.next = e
                e.next = last
                prelast = e
                i += 1
                e = edges[i]
            else:
                break
            if draw:
                if not clipline:
                    if keep_prelast.x > prelast.x:
                        x1, x2 = (prelast.x + delta) >> XY_SHIFT, keep_prelast.x >> XY_SHIFT
                    else:
                        x1, x2 = (keep_prelast.x + delta) >> XY_SHIFT, prelast.x >> XY_SHIFT
                    if x1 < width and x2 >= 0:
                        x1, x2 = max(x1, 0), min(x2, width - 1)
                        if x1 <= x2:
                            grid[y, x1:x2 + 1] = 1.0
                keep_prelast.x += keep_prelast.dx
                prelast.x += prelast.dx
            draw = not draw
        # the active list back in x order (OpenCV's bubble sort, exchanges and all)
        keep_prelast = None
        while True:
            prelast, last = head, head.next
            last_exchange = None
            while last is not keep_prelast and last.next is not None:
                te = last.next
                if last.x > te.x:
                    prelast.next = te
                    last.next = te.next
                    te.next = last
                    prelast = te
                    last_exchange = prelast
                else:
                    prelast, last = last, te
            if last_exchange is None:
                break
            keep_prelast = last_exchange
            if keep_prelast is head.next or keep_prelast is head:
                break


def fill_polygon(grid: np.ndarray, points) -> np.ndarray:
    """cv2.fillPoly(grid, [points], 1.0): points an (N, 2) array of integer grid coordinates; fills grid in place."""
    points = [(int(x), int(y)) for x, y in np.asarray(points)]
    if not points:
        return grid
    edges: list[_Edge] = []
    _collect_edges(grid, points, edges)
    _fill_edges(grid, edges)
    return grid


def point_in_polygon(points, x: float, y: float) -> bool:
    """cv2.pointPolygonTest(points (float32), (x, y), False) >= 0: inside or on an edge (crossing count)."""
    pts = np.asarray(points, np.float32)
    if len(pts) == 0:
        return False
    px, py = np.float32(x), np.float32(y)
    counter = 0
    vx, vy = pts[-1]
    for nx, ny in pts:
        v0x, v0y, vx, vy = vx, vy, nx, ny
        if (v0y <= py and vy <= py) or (v0y > py and vy > py) or (v0x < px and vx < px):
            if py == vy and (px == vx or (py == v0y and ((v0x <= px <= vx) or (vx <= px <= v0x)))):
                return True
            continue
        # (double)(pt.y - v0.y)*(v.x - v0.x) - (double)(pt.x - v0.x)*(v.y - v0.y): float32 differences, float64 products
        dist = float(py - v0y) * float(vx - v0x) - float(px - v0x) * float(vy - v0y)
        if dist == 0:
            return True
        if vy < v0y:
            dist = -dist
        counter += dist > 0
    return counter % 2 == 1

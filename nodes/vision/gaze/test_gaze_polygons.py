"""gaze_polygons.py, OpenCV's fillPoly and pointPolygonTest in numpy: against rasters OpenCV 5.0.0 made (always), and
against OpenCV itself where it is installed (envs/gaze: uv run --project envs/gaze --with pytest pytest nodes/vision/gaze).
Measured there besides: equal on 30000 random polygons (some reaching past the grid) and on all 8791 object outlines
of the bake-off's OOO and NES objects.json, rasterised as gaze_targets.Polygon does, and on 239767 point tests."""

import numpy as np
import pytest

from gaze_polygons import clip_line, fill_polygon, line_points, point_in_polygon

# cv2.fillPoly(np.zeros((14, 15), np.float32), [np.array(points, np.int32)], 1.0) with opencv-python 5.0.0.93
OPENCV = {
    "concave": ([[2, 2], [13, 3], [7, 7], [13, 12], [1, 11]], [
        "...............",
        "...............",
        "..######.......",
        "..############.",
        "..###########..",
        "..#########....",
        "..########.....",
        ".#######.......",
        ".########......",
        ".##########....",
        ".###########...",
        ".############..",
        "........######.",
        "...............",
    ]),
    "clipped": ([[-3, 4], [9, -2], [17, 6], [10, 15]], [
        ".....#######...",
        "...##########..",
        ".#############.",
        "###############",
        "###############",
        "###############",
        "###############",
        ".##############",
        "..#############",
        "...############",
        ".....#########.",
        "......#######..",
        ".......######..",
        "........####...",
    ]),
    "crossing": ([[1, 1], [13, 12], [13, 1], [1, 12]], [
        "...............",
        ".#...........#.",
        ".##.........##.",
        ".###.......###.",
        ".####.....####.",
        ".#####...#####.",
        ".#############.",
        ".#############.",
        ".#####...#####.",
        ".####.....####.",
        ".###.......###.",
        ".##.........##.",
        ".#...........#.",
        "...............",
    ]),
    "thin": ([[3, 2], [4, 2], [11, 12]], [
        "...............",
        "...............",
        "...##..........",
        "....##.........",
        ".....#.........",
        ".....##........",
        "......##.......",
        ".......#.......",
        "........#......",
        ".........#.....",
        ".........##....",
        "..........#....",
        "...........#...",
        "...............",
    ]),
}


@pytest.mark.parametrize("name", sorted(OPENCV))
def test_a_polygon_fills_as_opencvs_fillpoly_does(name):
    points, rows = OPENCV[name]
    grid = fill_polygon(np.zeros((14, 15), np.float32), np.array(points))
    assert ["".join("#" if v else "." for v in row) for row in grid] == rows


def test_lines_are_8_connected_from_left_to_right_and_clipped_to_the_grid():
    assert line_points(10, 10, (0, 0), (3, 1)) == [(0, 0), (1, 0), (2, 1), (3, 1)]
    assert line_points(10, 10, (3, 1), (0, 0)) == line_points(10, 10, (0, 0), (3, 1))   # drawn left to right either way
    assert line_points(10, 10, (5, -5), (5, 2)) == [(5, 0), (5, 1), (5, 2)]
    assert line_points(10, 10, (-5, -5), (-1, 20)) == []
    assert clip_line(10, 10, (-3, 4), (9, -2)) == (True, (0, 3), (5, 0))   # cv2.clipLine: its truncating arithmetic


def test_a_point_is_in_a_polygon_inside_or_on_its_outline_as_pointpolygontest_says():
    """cv2.pointPolygonTest(geom, p, False) of opencv-python 5.0.0.93: 0 (on the outline) and 1 count, -1 does not."""
    geom = np.array([[100, 50], [300, 80], [250, 300], [120, 260]], np.float32)
    on, inside, outside = [(100, 50), (200, 65), (110, 155)], [(180, 150)], [(99.5, 50), (300, 81)]
    assert [point_in_polygon(geom, *p) for p in on + inside + outside] == [True] * 4 + [False] * 2


def test_against_opencv_itself_where_it_is_installed():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    for k in range(500):
        points = rng.integers(-40, 300, size=(int(rng.integers(3, 41)), 2))
        want = np.zeros((256, 256), np.float32)
        cv2.fillPoly(want, [points.astype(np.int32)], 1.0)
        assert np.array_equal(fill_polygon(np.zeros((256, 256), np.float32), points), want), points.tolist()
        geom = rng.uniform(-10, 100, size=(int(rng.integers(3, 12)), 2)).astype(np.float32)
        x, y = rng.uniform(-10, 100, 2) if k % 3 else (float(geom[0, 0]), float(geom[0, 1]))
        assert point_in_polygon(geom, x, y) == (cv2.pointPolygonTest(geom.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0)

"""Face crops (frames.py), with nothing on PATH: the app runs nodes from the Finder, where ffmpeg is not found.
Run: uv run --project envs/faces --with pytest pytest nodes/vision/face-track"""

import numpy as np
import pytest
from PIL import Image

from frames import crop_box, crop_pixels, longest_side, save_crop, square, width


@pytest.fixture(autouse=True)
def no_tools_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "")


def test_crop_box_is_padded_clipped_and_even():
    assert crop_box([10, 10, 20, 20], 0.25, 64, 48) == (4, 4, 30, 30)    # 5..35 -> even
    assert crop_box([50, 30, 20, 20], 0.5, 64, 48) == (40, 20, 24, 28)   # clipped to the frame


def test_sizes_are_ffmpegs():
    assert longest_side(256)(30, 20) == (256, 170)   # scale='if(gt(iw,ih),256,-2)':'if(gt(iw,ih),-2,256)'
    assert longest_side(256)(20, 30) == (170, 256)
    assert width(320)(30, 21) == (320, 224)          # scale=320:-2
    assert square(64)(30, 20) == (64, 64)


def test_crop_pixels(clip):
    pixels = crop_pixels(clip.path, 0.4, [8, 8, 16, 16], 0.0, 64)
    assert pixels.shape == (64, 64, 3) and pixels.dtype == np.uint8
    assert abs(pixels.mean() - clip.nth(4)[8:24, 8:24].mean()) < 1


def test_save_crop(clip, tmp_path):
    save_crop(clip.path, 1.2, [8, 8, 30, 20], 0.0, tmp_path / "c.jpg", longest_side(256))
    img = Image.open(tmp_path / "c.jpg")
    assert img.size == (256, 170)
    assert abs(np.asarray(img).mean() - clip.nth(12)[8:28, 8:38].mean()) < 1


def test_a_crop_is_cut_from_the_upright_frame_the_detector_saw(rotated):
    # torchcodec, which reads the detector's frames, turns a rotated clip upright: 32x64 here, so its boxes are in that
    # frame. The crop of the bottom half is red, of the top half blue, and a box past the stored 64x32 still fits.
    bottom = crop_pixels(rotated, 0.5, [0, 40, 32, 24], 0.0, 16).astype(int).mean((0, 1))
    top = crop_pixels(rotated, 0.5, [0, 0, 32, 24], 0.0, 16).astype(int).mean((0, 1))
    assert bottom[0] > 200 > bottom[2] and top[2] > 200 > top[0], (bottom, top)

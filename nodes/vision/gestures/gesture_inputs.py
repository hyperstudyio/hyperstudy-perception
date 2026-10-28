"""What the gesture rules read of the other layers (gesture_rules.sample_evidence's `ctx`), checked as it is read: the
hands layer must be RTMPose-hand's (its keypoint threshold is that model's), and every hand's body and face, and every
face track's landmarks, must be in the bodies, faces and faces_detail the run gave. numpy only, apart from
upright_size (PyAV, which reads one frame of the video).
"""

from __future__ import annotations

import numpy as np

from gesture_rules import HAND_MODEL, Series, face_geometry


class InputError(ValueError):
    """The layers do not belong together, or are not what the rules were made for (the node's NodeError("input"))."""


def upright_size(video) -> tuple[int, int]:
    """(width, height) of the video's frames as players show them, the coordinates the hands layer's keypoints are in
    (body-pose and hands decode them upright): a phone video stored sideways with a 90 or 270 degree rotation is as
    tall as its stored width. media.v1's width and height are the stored ones."""
    import av
    with av.open(str(video)) as container:
        frame = next(container.decode(container.streams.video[0]), None)
        if frame is None:
            raise InputError("the video has no frames")
        quarter_turns = round((frame.rotation or 0) / 90) % 4
        return (frame.height, frame.width) if quarter_turns % 2 else (frame.width, frame.height)


def check_hands_model(hands_layer: dict) -> None:
    models = [m.get("id", "") for m in hands_layer.get("producer", {}).get("models", [])]
    if not any(HAND_MODEL in m for m in models):
        raise InputError(f"the hands layer's keypoints are not {HAND_MODEL}'s (its models: {models}): the keypoint "
                         "score threshold is that model's")


class Context:
    """Lookups by time into the bodies, faces, faces_detail and (optional) objects layers, and the frame size."""

    def __init__(self, hands: list[dict], bodies: dict, faces: dict, detail: dict, objects: dict | None,
                 width: int, height: int):
        self.width, self.height = width, height
        names = bodies["keypoint_names"]
        wanted = ("nose", "left_shoulder", "right_shoulder")
        if missing := [n for n in wanted if n not in names]:
            raise InputError(f"the bodies layer has no {', '.join(missing)} keypoint")
        self.body_idx = {n: names.index(n) for n in wanted}
        self.body_threshold = bodies["score_threshold"]
        self.bodies = {tr["id"]: {round(s["t"], 3): np.asarray(s["kpts"], dtype=float) for s in tr["samples"]}
                       for tr in bodies["tracks"]}
        self.faces = [(tr["id"], tr["shot"], Series(tr["samples"], "box")) for tr in faces["tracks"]]
        self.face_box = {face: series for face, _, series in self.faces}
        missing = [face for face in self.face_box if face not in detail["tracks"]]
        if missing:
            raise InputError(f"faces_detail has no landmarks for {len(missing)} face tracks (e.g. {missing[:3]}): it is not "
                             "this faces layer's")
        self.landmarks = {face: Series(detail["tracks"][face], "lmk") for face in self.face_box}
        self.objects = [(tr["id"], tr["shot"], Series(tr["samples"], "box")) for tr in (objects or {}).get("tracks", [])]
        self.hand_boxes = {}   # (body, side, t) -> box: the same body's other hand, for the clasped-hands check
        for hand in hands:
            if hand["body"] is not None and hand["body"] not in self.bodies:
                raise InputError(f"{hand['id']}'s body {hand['body']} is not in the bodies layer")
            if hand["face"] is not None and hand["face"] not in self.face_box:
                raise InputError(f"{hand['id']}'s face {hand['face']} is not in the faces layer")
            if hand["body"] is not None:
                for s in hand["samples"]:
                    self.hand_boxes[(hand["body"], hand["side"], s["t"])] = s["box"]

    def hand_box(self, body: str, side: str | None, t: float):
        return self.hand_boxes.get((body, side, t))

    def body_at(self, body: str, t: float):
        kpts = self.bodies[body].get(round(t, 3))
        if kpts is None:
            raise InputError(f"{body} has no sample at {t} s: the hands layer is not this bodies layer's")
        return kpts

    def face_geometry(self, face: str, t: float):
        """(mouth centre, face size, face region) of the face track at t, or None without landmarks or box there."""
        lmk, box = self.landmarks[face].at(t), self.face_box[face].at(t)
        return None if lmk is None or box is None else face_geometry(lmk, box)

    def targets(self, t: float, shot: str, own_face: str | None) -> list[tuple[str, np.ndarray]]:
        """(id, box) of what a hand of own_face in this shot can point at, at t: the shot's other face tracks, then its
        object tracks (none without the objects layer)."""
        out = []
        for ident, track_shot, series in self.faces + self.objects:
            if track_shot == shot and ident != own_face:
                box = series.at(t)
                if box is not None:
                    out.append((ident, box))
        return out

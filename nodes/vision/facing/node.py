"""vision.facing: who faces whom, as the bundle's facing layer: events in which one face track's body is turned toward
another's, and whether the other faces back.

The people are the bodies layer's (YOLOX-m boxes, RTMW keypoints, linked to face tracks by body-pose): one person-box
pass for every node that needs people (PLAN.md §4), where the bake-off ran YOLOX again for this. Facing compares only
bodies linked to a face track, so only their samples are posed. For each, MediaPipe Pose Landmarker heavy poses a crop
around the body box (models.py) and facing.py turns its world landmarks into a torso direction, and the directions at
each sample time into events. Frames are decoded in order with OpenCV, as body-pose read them.

A body whose crop MediaPipe finds no pose in takes no part at that time, neither facing nor faced: as in the bake-off,
whose bodies were the ones MediaPipe posed.

Beside the layer, two side files keep what it drops (facing_raw.py): layers/facing_poses.parquet, every pose MediaPipe
returned for every crop (all 33 landmarks with visibility and presence, the world landmarks, the torso direction and
centre, and which pose was kept), and layers/facing_angles.parquet, every pair's angle and distance at every sample
time, from which the events can be thresholded again (cone_deg, min_facing_s, max_gap_s). The node reads the angles file
back and checks that thresholding it at the configured values gives the layer's events. The layer points at them with
raw.poses and raw.angles.

Licence: MediaPipe's code and the Pose Landmarker model are Apache-2.0 (commercial-ok), but the people come from the
bodies layer (YOLOX-m and RTMW, trained on non-commercial data) and the events name py-feat's face tracks: the layer is
academic-only through its inputs.
"""

import time

import cv2

from perception_node import Node, NodeError
import facing_raw
from facing import POSES, body_poses, crop, events_from_angles, linked_samples, pair_angles
from models import LICENCE, MODEL_ID, WEIGHTS, Landmarker


def frame_indices(samples, fps: float) -> dict[float, int]:
    """{sample time: frame index}: body-pose sampled frame i at t = round(i / fps, 3)."""
    indices = {}
    for _, t, *_ in samples:
        i = round(t * fps)
        if round(i / fps, 3) != t:
            raise NodeError("input", f"the bodies layer has a sample at {t} s, which is no frame of this {fps} fps video")
        indices[t] = i
    return indices


def frames_at(video, indices):
    """(frame index, BGR frame) for the sorted frame indices, decoded in order."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise NodeError("input", f"OpenCV cannot open {video.name}")
    wanted = iter(sorted(set(indices)))
    target = next(wanted, None)
    index = 0
    try:
        while target is not None:
            if index < target:
                if not capture.grab():
                    break
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                yield index, frame
                target = next(wanted, None)
            index += 1
    finally:
        capture.release()
    if target is not None:
        raise NodeError("input", f"{video.name} ended at frame {index}, before frame {target}, which has body samples")


with Node() as node:
    c = node.config
    video, media, bodies = node.input("video"), node.input_json("media"), node.input_json("bodies")
    samples = linked_samples(bodies)
    at_time = frame_indices(samples, media["fps"])
    by_index = {}
    for sample in samples:
        by_index.setdefault(at_time[sample[1]], []).append(sample)

    t0 = time.time()
    landmarker = Landmarker(node.accelerator, POSES)
    load_s = round(time.time() - t0, 1)
    node.model(MODEL_ID, licence_mode="commercial-ok", source=WEIGHTS["pose_landmarker_heavy"]["url"],
               revision=WEIGHTS["pose_landmarker_heavy"]["sha256"], licence=LICENCE)

    frames = {}   # (shot, t) -> [(face number, torso centre, [yaw, pitch] or None)]
    counts = {"torso": 0, "shoulders": 0, "no_facing": 0, "no_pose": 0}
    pose_s, t_all = 0.0, time.time()
    try:
        with facing_raw.PosesWriter(node.path("poses", "facing_poses.parquet"), MODEL_ID) as poses_file:
            for index, image in frames_at(video, by_index):
                for shot, t, face, box, body, sample in by_index[index]:
                    square, window = crop(image, box)
                    tick = time.time()
                    raw = landmarker(square)
                    pose_s += time.time() - tick
                    poses, kept = body_poses(raw, window, box)
                    for pose in poses:
                        poses_file.add(body, sample, t, face, window, pose)
                    if kept is None:
                        counts["no_pose"] += 1
                        continue
                    centre, facing, source = kept
                    counts[source or "no_facing"] += 1
                    frames.setdefault((shot, t), []).append((face, centre, facing))
        landmarker.close()
        angles = pair_angles(frames)
        events = events_from_angles(angles, c["cone_deg"], c["min_facing_s"], c["max_gap_s"])
        angles_path = node.path("angles", "facing_angles.parquet")
        facing_raw.write_angles(angles_path, angles, c["cone_deg"], c["min_facing_s"], c["max_gap_s"])
    except facing_raw.BadRow as e:
        raise NodeError("internal", f"the model's output: {e}") from None
    # the angles file is what re-thresholding reads: at the configured values it must give the layer's events
    if events_from_angles(facing_raw.read_angles(angles_path), c["cone_deg"], c["min_facing_s"], c["max_gap_s"]) != events:
        raise NodeError("internal", "thresholding the stored angles does not give the layer's events")
    total_s = round(time.time() - t_all, 1)

    node.write_json("facing", "facing.json", {
        "layer": "facing", "schema_version": "0.1.0",
        "producer": node.producer(timings={"load_s": load_s, "pose_s": round(pose_s, 1), "total_s": total_s}, device="cpu-xnnpack"),
        "events": events,
        "raw": {"poses": {"file": facing_raw.POSES_FILE, "format": "parquet", "description": facing_raw.POSES_DESCRIPTION},
                "angles": {"file": facing_raw.ANGLES_FILE, "format": "parquet", "description": facing_raw.ANGLES_DESCRIPTION}}})
    node.usage(frames=len(by_index), bodies=len(samples), events=len(events), mutual=sum(e["mutual"] for e in events),
               faces=len({e["from"] for e in events} | {e["to"] for e in events}), facing_from=counts,
               poses=poses_file.rows, angles=len(angles),
               realtime_factor=round(total_s / media["duration"], 3) if media.get("duration") else None)

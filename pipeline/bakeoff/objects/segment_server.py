"""Click-to-segment for the annotator: one point in, one object track (within that shot) out.

The annotator page (served from http://127.0.0.1:8766) posts a click; the server finds
the shot containing t, decodes it at OBJECT_FPS, gives SAM 3's tracker (facebook/sam3,
Sam3TrackerVideoModel, i.e. SAM 3 with a point prompt instead of text) a positive point
on the nearest frame, and propagates the mask forward and backward to the shot's ends.
A track never crosses a cut.

The model is loaded once at startup. The decoded shot and its image features are kept
for the most recent shot, so a second click in the same shot skips decoding and the
image encoder. Requests are handled one at a time (the model is not thread-safe).

Request:  POST /segment  {"run": "runs/nes-biking", "t": 12.4, "x": 640, "y": 300}
          run is relative to the bake-off directory; x, y are full-resolution video pixels.
Response: {"label": null, "start": 11.43, "end": 12.387,
           "samples": [{"t", "box": [x, y, w, h], "poly": [[x, y], ...], "score"}]}
          (the same sample shape as objects.json). 400 for a bad request, 422 when SAM
          finds no object at the point.

CORS allows only the annotator origin. Binds to 127.0.0.1 only.

On an NVIDIA machine pass --device cuda (bf16, as in objects.py; not yet run on CUDA). So far it
has been tested on CPU only (responses, CORS, validation): a first click in a shot took 36.5 s and a
repeat click in the same shot 2.6 s for a 6-frame shot. Latency on MPS and CUDA is not measured.

Usage (from objects/): uv run python segment_server.py [--device mps|cuda|cpu] [--port 8767]
  curl -s -X POST http://127.0.0.1:8767/segment -H 'Content-Type: application/json' \\
       -d '{"run": "runs/nes-biking", "t": 40.0, "x": 960, "y": 540}'
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from objects import BAKEOFF, OBJECT_FPS, SEGMENTER_DTYPE, frame_sample, load_env, shot_frames

HOST = "127.0.0.1"
ALLOWED_ORIGIN = "http://127.0.0.1:8766"  # the annotator page
TRACKER = "facebook/sam3"


class Segmenter:
    def __init__(self, device, fps, dtype=None):
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        self.device, self.fps = torch.device(device), fps
        self.dtype = dtype or SEGMENTER_DTYPE[self.device.type]  # bf16 on MPS, as in objects.py
        self.model = Sam3TrackerVideoModel.from_pretrained(TRACKER, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(TRACKER)
        self.cached = None  # (run, shot index) -> session with every frame's image features kept

    def session_for(self, run, shot, video_meta):
        key = (run, shot["index"])
        if self.cached and self.cached[0] == key:
            session, times = self.cached[1], self.cached[2]
            session.reset_tracking_data()  # keeps decoded frames and cached image features
            return session, times
        self.cached = None
        frames, times = shot_frames(BAKEOFF / video_meta["source"], shot["start"], shot["end"], self.fps,
                                    video_meta["width"], video_meta["height"])
        session = self.processor.init_video_session(
            video=list(frames), inference_device=self.device, processing_device="cpu", video_storage_device="cpu",
            max_vision_features_cache_size=len(frames), dtype=self.dtype)
        self.cached = (key, session, times)
        return session, times

    def masks(self, session, output):
        masks = self.processor.post_process_masks(
            [output.pred_masks], original_sizes=[[session.video_height, session.video_width]], binarize=True)[0]
        return masks[0, 0].cpu().numpy(), float(torch.sigmoid(output.object_score_logits.float()).flatten()[0])

    @torch.inference_mode()
    def segment(self, run, t, x, y):
        run_dir = (BAKEOFF / run).resolve()
        if not run_dir.is_relative_to((BAKEOFF / "runs").resolve()) or not (run_dir / "extract.json").exists():
            raise ValueError(f"unknown run {run!r}")
        extract = json.loads((run_dir / "extract.json").read_text())
        meta = extract["video"]
        if not (0 <= x < meta["width"] and 0 <= y < meta["height"]):
            raise ValueError(f"point ({x}, {y}) outside the {meta['width']}x{meta['height']} frame")
        shot = next((s for s in extract["shots"] if s["start"] <= t < s["end"]), None)
        if shot is None:
            raise ValueError(f"t={t} is outside the video")

        session, times = self.session_for(run, shot, meta)
        frame_idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=frame_idx, obj_ids=1,
            input_points=[[[[float(x), float(y)]]]], input_labels=[[[1]]])
        self.model(inference_session=session, frame_idx=frame_idx)

        by_frame = {}
        for reverse in (False, True):
            for output in self.model.propagate_in_video_iterator(session, start_frame_idx=frame_idx, reverse=reverse):
                mask, score = self.masks(session, output)
                if score > 0.5 and mask.any():  # object_score_logits > 0: SAM says the object is visible
                    by_frame[output.frame_idx] = frame_sample(times[output.frame_idx], mask, score)
        samples = [by_frame[i] for i in sorted(by_frame)]
        return {"label": None, "start": samples[0]["t"] if samples else None,
                "end": samples[-1]["t"] if samples else None, "samples": samples}


def make_handler(segmenter):
    class Handler(BaseHTTPRequestHandler):
        def cors(self):
            if self.headers.get("Origin") == ALLOWED_ORIGIN:
                self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
                self.send_header("Vary", "Origin")

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            if self.path != "/segment" or self.headers.get("Origin") != ALLOWED_ORIGIN:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(204)
            self.cors()
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            if self.path != "/segment":
                return self.reply(404, {"error": "not found"})
            origin = self.headers.get("Origin")
            if origin is not None and origin != ALLOWED_ORIGIN:
                return self.reply(403, {"error": f"origin {origin} not allowed"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                run, t, x, y = str(body["run"]), float(body["t"]), float(body["x"]), float(body["y"])
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": f"expected JSON {{run, t, x, y}}: {exc}"})
            t0 = time.time()
            try:
                result = segmenter.segment(run, t, x, y)
            except ValueError as exc:
                return self.reply(400, {"error": str(exc)})
            if not result["samples"]:
                return self.reply(422, {"error": "no object found at that point"})
            self.log_message("segment %s t=%.2f (%d, %d): %d samples, %.1f s", run, t, x, y,
                             len(result["samples"]), time.time() - t0)
            self.reply(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--fps", type=float, default=OBJECT_FPS)
    args = parser.parse_args()
    load_env()
    t0 = time.time()
    segmenter = Segmenter(args.device, args.fps)
    print(f"loaded {TRACKER} on {args.device} in {time.time() - t0:.1f} s; listening on http://{HOST}:{args.port}",
          flush=True)
    HTTPServer((HOST, args.port), make_handler(segmenter)).serve_forever()


if __name__ == "__main__":
    main()

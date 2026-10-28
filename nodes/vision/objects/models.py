"""SAM 3 (segmentation and tracking with text prompts) and DINOv3 ViT-L/16 (embeddings for cross-shot linking), as the
bake-off ran them (tools/perception-bakeoff/objects/objects.py): through transformers 5.20.2, from the Hugging Face
hub at pinned commits. Both are gated: the account whose HF_TOKEN the node is given must have accepted Meta's terms at
https://hf.co/facebook/sam3 and https://hf.co/facebook/dinov3-vitl16-pretrain-lvd1689m.

Where it runs (the request's accelerator): mps (the Apple GPU), cuda, or cpu. SAM 3 runs in bfloat16 on a GPU (on MPS
the bake-off measured it 2.5x faster than float32 with the same objects on every frame of two NES shots, masks at
median IoU 0.998), float32 on the CPU; DINOv3 in float32 everywhere. A device PyTorch cannot use here fails the node; it
never runs elsewhere instead.

SAM 3's post-processing (NMS, hole filling, sprinkle removal) comes in transformers from a CUDA-only hub kernel
(kernels-community/cv-utils); without it Sam3VideoModel silently skips all three. CpuCvUtils puts in its place the CPU
implementations from Meta's reference code (facebookresearch/sam3 perflib), on every device, so the post-processing SAM
3 was designed with runs here and results match across machines (the bake-off's choice; the CUDA path has not been
run, there or here).
"""

from __future__ import annotations

from perception_node import NodeError

SEGMENTER = "facebook/sam3"
SEGMENTER_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
EMBEDDER = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EMBEDDER_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
SEGMENTER_LICENCE = ("SAM License (Meta; gated on Hugging Face): commercial use permitted; acknowledge use in publications; "
                     "trade-control, no-reverse-engineering and indemnity terms")
EMBEDDER_LICENCE = ("DINOv3 License (Meta; gated on Hugging Face): commercial use permitted; acknowledge use in publications; "
                    "redistribution only under the same licence with a copy of it; trade-control, no-reverse-engineering and "
                    "indemnity terms; Meta may amend the terms")
EMBED_MEAN, EMBED_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet, as DINOv3's processor uses


def device_for(accelerator: str):
    """The torch device for the request's accelerator, or NodeError when PyTorch cannot use it here."""
    import torch
    if accelerator == "mps" and not torch.backends.mps.is_available():
        raise NodeError("resources", "PyTorch cannot use the Apple GPU (MPS) here: run Objects on another device")
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) finds no usable CUDA GPU here: "
                                     "run Objects on another device")
    return torch.device(accelerator if accelerator in ("mps", "cuda") else "cpu")


def segmenter_dtype(device):
    import torch
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def synchronize(device):
    """Wait for queued GPU work, so per-shot timings measure the work and not just its submission."""
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class CpuCvUtils:
    """Stand-in for the CUDA-only kernels-community/cv-utils kernel, from Meta's sam3 perflib CPU paths."""

    @staticmethod
    def generic_nms(ious, scores, iou_threshold, use_iou_matrix=True):
        import torch
        ious_np = ious.float().cpu().numpy()
        order = scores.float().cpu().numpy().argsort()[::-1]
        kept = []
        while order.size > 0:
            i = order[0]
            kept.append(i)
            order = order[1:][ious_np[i, order[1:]] <= iou_threshold]
        return torch.tensor(kept, dtype=torch.int64, device=scores.device)

    @staticmethod
    def cc_2d(mask, get_counts=True):
        """8-connected components per (N, 1, H, W) mask; counts = size of each pixel's component (0 off-mask)."""
        import cv2
        import numpy as np
        import torch
        masks = mask.cpu().numpy().astype(np.uint8)
        labels_out = np.zeros(masks.shape, np.int32)
        counts_out = np.zeros(masks.shape, np.int32)
        for i in range(masks.shape[0]):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(masks[i, 0], connectivity=8)
            areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
            areas[0] = 0   # label 0 is the off-mask region
            labels_out[i, 0] = labels
            counts_out[i, 0] = areas[labels]
        return torch.from_numpy(labels_out).to(mask.device), torch.from_numpy(counts_out).to(mask.device)


def _from_hub(load, repo: str):
    """load() with Hugging Face's gating and network failures as node errors."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError
    try:
        return load()
    except GatedRepoError:   # before RepositoryNotFoundError, which it is a kind of; also what an invalid token gets
        raise NodeError("licence", f"{repo} is gated, and Hugging Face refused this machine's HF_TOKEN for it: check the token, "
                                   f"and that its account accepted the conditions at https://hf.co/{repo}") from None
    except RepositoryNotFoundError:
        raise NodeError("input", f"Hugging Face did not accept this machine's HF_TOKEN for {repo}") from None
    except (HfHubHTTPError, LocalEntryNotFoundError) as e:   # the network or the hub failed, and it is not in the cache
        raise NodeError("transient", f"could not download {repo} from Hugging Face: {e}") from None


def load_segmenter(device, token: str | None):
    """SAM 3 video (detector + tracker) and its processor, with the CPU post-processing installed."""
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.models.sam3_video import modeling_sam3_video
    modeling_sam3_video.cv_utils_kernel = CpuCvUtils()

    def load():
        model = Sam3VideoModel.from_pretrained(SEGMENTER, revision=SEGMENTER_REVISION, token=token, dtype=segmenter_dtype(device))
        return model.to(device).eval(), Sam3VideoProcessor.from_pretrained(SEGMENTER, revision=SEGMENTER_REVISION, token=token)
    return _from_hub(load, SEGMENTER)


def check_access(token: str | None):
    """Both gated models' config at their pinned revisions, from the Hugging Face cache or the hub: done before the
    node uploads or spends anything, so a token Hugging Face refuses fails the node first."""
    from huggingface_hub import hf_hub_download
    for repo, revision in ((SEGMENTER, SEGMENTER_REVISION), (EMBEDDER, EMBEDDER_REVISION)):
        _from_hub(lambda: hf_hub_download(repo, "config.json", revision=revision, token=token), repo)


def load_embedder(device, token: str | None):
    from transformers import AutoModel
    return _from_hub(lambda: AutoModel.from_pretrained(EMBEDDER, revision=EMBEDDER_REVISION, token=token).to(device).eval(), EMBEDDER)


def segment_shot(segmenter, frames: list, times, nouns, device, frame_sample) -> dict:
    """SAM 3 over one shot, all its nouns as text prompts: {SAM 3 object id: {label, score, samples}}. A session never
    spans a cut; SAM 3 detects on every frame and tracks with memory, so objects entering mid-shot are picked up.
    Empties `frames` once the session holds its own copy: the RGB frames are not needed for the pass (budget.py)."""
    import torch
    model, processor = segmenter
    with torch.inference_mode():
        session = processor.init_video_session(video=frames, inference_device=device, processing_device="cpu",
                                               video_storage_device="cpu", dtype=segmenter_dtype(device))
        frames.clear()
        processor.add_text_prompt(session, nouns)
        tracks = {}
        for out in model.propagate_in_video_iterator(inference_session=session):
            post = processor.postprocess_outputs(session, out)
            prompt_of = {obj: text for text, objs in post["prompt_to_obj_ids"].items() for obj in objs}
            masks = post["masks"].cpu().numpy()
            for obj_id, det_score, mask in zip(post["object_ids"].tolist(), post["scores"].tolist(), masks):
                if not mask.any():
                    continue
                s = frame_sample(times[out.frame_idx], mask, out.obj_id_to_tracker_score.get(obj_id, det_score))
                track = tracks.setdefault(obj_id, {"label": prompt_of[obj_id], "score": det_score, "samples": []})
                track["samples"].append(s)
        del session
    return tracks


def embed_crops(model, crops, device):
    """Crops are EMBED_PX squares (geometry.object_views): (normalised mean CLS embedding, normalised per-crop CLS)."""
    import numpy as np
    import torch
    with torch.inference_mode():
        batch = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float() / 255
        mean, std = (torch.tensor(v, device=device)[None, :, None, None] for v in (EMBED_MEAN, EMBED_STD))
        feats = model(pixel_values=(batch - mean) / std).pooler_output
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return torch.nn.functional.normalize(feats.mean(0), dim=-1).cpu().numpy(), feats.cpu().numpy()

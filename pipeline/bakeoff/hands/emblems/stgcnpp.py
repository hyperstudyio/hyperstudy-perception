"""PYSKL's light ST-GCN++ for hand gestures (demo/stgcnpp_gesture.py + demo/hagrid.pth) in plain PyTorch.

Extracted from PYSKL (https://github.com/kennymckormick/pyskl, commit f2bf3a6, Apache-2.0, Copyright OpenMMLab) so
the checkpoint loads without mmcv: pyskl/utils/graph.py (Graph, layout 'handmp', mode 'spatial'),
pyskl/models/gcns/utils/gcn.py (unit_gcn), pyskl/models/gcns/utils/tcn.py (unit_tcn, mstcn),
pyskl/models/gcns/stgcn.py (STGCNBlock, STGCN), pyskl/models/heads/simple_head.py (GCNHead, mode 'GCN') and
RecognizerGCN.forward_test with average_clips 'prob'. Only the code paths the gesture config uses are kept; mmcv's
build_norm_layer('BN') and build_activation_layer('ReLU') are nn.BatchNorm2d and nn.ReLU. Module names match the
checkpoint's state-dict keys, so it loads with strict=True.

The config (demo/stgcnpp_gesture.py): graph 'handmp', joint modality, in_channels 2, gcn_adaptive 'init',
gcn_with_res True, tcn_type 'mstcn', num_stages 6, down/inflate stage 6, GCNHead(num_classes=40, in_channels=128).
Test pipeline: PreNormalize2D(threshold=0, mode='auto'), GenSkeFeat(j), UniformSample(clip_len=10), PoseDecode,
FormatGCNInput(num_person=1); preprocess() below reproduces it for one hand.
"""

import numpy as np
import torch
import torch.nn as nn

# GestureDataset.label_names (pyskl/datasets/gesture_dataset.py). Indices 0-26 are 20BN-Jester's classes with Thumb
# Down/Up renamed Dislike/Like, 27-39 HaGRID's. emblems.py documents which of them hagrid.pth was trained on.
LABEL_NAMES = [
    'Doing other things', 'Drumming Fingers', 'No gesture',
    'Pulling Hand In', 'Pulling Two Fingers In', 'Pushing Hand Away', 'Pushing Two Fingers Away',
    'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand',
    'Sliding Two Fingers Down', 'Sliding Two Fingers Left',
    'Sliding Two Fingers Right', 'Sliding Two Fingers Up',
    'Stop Sign', 'Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up',
    'Dislike', 'Like', 'Turning Hand Clockwise', 'Turning Hand Counterclockwise',
    'Zooming In With Full Hand', 'Zooming In With Two Fingers',
    'Zooming Out With Full Hand', 'Zooming Out With Two Fingers',
    'Call', 'Fist', 'Four', 'Mute', 'OK', 'One', 'Palm',
    'Peace', 'Rock', 'Three-Middle', 'Three-Left', 'Two Up', 'No Gesture',
]
CLIP_LEN = 10  # UniformSample(clip_len=10)
HANDMP_INWARD = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9), (11, 10), (12, 11),
                 (13, 0), (14, 13), (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)]
NUM_NODE = 21


def _edge2mat(link, num_node):
    a = np.zeros((num_node, num_node))
    for i, j in link:
        a[j, i] = 1
    return a


def _normalize_digraph(a, dim=0):
    dl = np.sum(a, dim)
    dn = np.zeros((a.shape[1], a.shape[1]))
    for i in range(a.shape[1]):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-1)
    return np.dot(a, dn)


def spatial_graph():
    """Graph(layout='handmp', mode='spatial').A: identity, inward, outward (3 x 21 x 21)."""
    self_link = [(i, i) for i in range(NUM_NODE)]
    outward = [(j, i) for (i, j) in HANDMP_INWARD]
    return np.stack((_edge2mat(self_link, NUM_NODE), _normalize_digraph(_edge2mat(HANDMP_INWARD, NUM_NODE)),
                     _normalize_digraph(_edge2mat(outward, NUM_NODE))))


class UnitGCN(nn.Module):
    """unit_gcn with adaptive='init', conv_pos='pre', with_res=True."""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_subsets = A.size(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.A = nn.Parameter(A.clone())
        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x)
        x = self.conv(x).view(n, self.num_subsets, -1, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        return self.act(self.bn(x) + res)


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm=True):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))  # dropout 0 in this config


class MSTCN(nn.Module):
    """mstcn with the default ms_cfg [(3,1),(3,2),(3,3),(3,4),('max',3),'1x1']."""

    MS_CFG = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        num_branches = len(self.MS_CFG)
        mid = out_channels // num_branches
        rem = out_channels - mid * (num_branches - 1)
        act = nn.ReLU()
        branches = []
        for i, cfg in enumerate(self.MS_CFG):
            bc = rem if i == 0 else mid
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, bc, kernel_size=1, stride=(stride, 1)))
            elif cfg[0] == 'max':
                branches.append(nn.Sequential(nn.Conv2d(in_channels, bc, kernel_size=1), nn.BatchNorm2d(bc), act,
                                              nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
            else:
                branches.append(nn.Sequential(nn.Conv2d(in_channels, bc, kernel_size=1), nn.BatchNorm2d(bc), act,
                                              UnitTCN(bc, bc, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=False)))
        self.branches = nn.ModuleList(branches)
        tin = mid * (num_branches - 1) + rem
        self.transform = nn.Sequential(nn.BatchNorm2d(tin), act, nn.Conv2d(tin, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.transform(torch.cat([b(x) for b in self.branches], dim=1)))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class STGCN(nn.Module):
    """STGCN(in_channels=2, base_channels=64, num_stages=6, inflate_stages=[6], down_stages=[6], data_bn_type='VC')."""

    def __init__(self, in_channels=2, base_channels=64, num_stages=6, inflate_stages=(6,), down_stages=(6,)):
        super().__init__()
        A = torch.tensor(spatial_graph(), dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)]
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            cin = base_channels
            if i in inflate_stages:
                inflate_times += 1
            base_channels = int(64 * 2 ** inflate_times + 1e-4)
            modules.append(STGCNBlock(cin, base_channels, A.clone(), stride))
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        for block in self.gcn:
            x = block(x)
        return x.reshape((n, m) + x.shape[1:])


class GCNHead(nn.Module):
    def __init__(self, num_classes=40, in_channels=128):
        super().__init__()
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        n, m, c, t, v = x.shape
        x = nn.functional.adaptive_avg_pool2d(x.reshape(n * m, c, t, v), 1).reshape(n, m, c).mean(dim=1)
        return self.fc_cls(x)


class GestureSTGCNpp(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = STGCN()
        self.cls_head = GCNHead()

    @torch.no_grad()
    def predict(self, keypoint):
        """keypoint: (batch, 1 clip, 1 person, T, 21, 2) as the pipeline formats it -> (batch, 40) probabilities,
        RecognizerGCN.forward_test with average_clips='prob'."""
        bs, nc = keypoint.shape[:2]
        scores = self.cls_head(self.backbone(keypoint.reshape((bs * nc,) + keypoint.shape[2:])))
        return torch.softmax(scores.reshape(bs, nc, -1), dim=2).mean(dim=1)


def load(checkpoint):
    model = GestureSTGCNpp()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.eval()


def preprocess(frames):
    """PYSKL's test pipeline for one hand: frames is (T, 21, 2) float keypoints -> (1, 1, CLIP_LEN, 21, 2).

    PreNormalize2D(mode='auto', threshold=0) without scores rescales by the clip's min/max ONLY when both extents
    exceed 10 (pixel coordinates); on [0, 1] image-normalised coordinates (what the demo and HaGRID's MediaPipe
    landmarks are) it is a no-op, so it is reproduced exactly. UniformSample in test mode with T < CLIP_LEN takes
    frames arange(CLIP_LEN) mod T (a single frame is repeated 10 times); T >= CLIP_LEN uses its seeded sampling,
    which is not reproduced here: callers pass at most CLIP_LEN frames.
    """
    kp = np.asarray(frames, dtype=np.float32)[None]  # M=1, T, V, C
    if kp.shape[1] > CLIP_LEN:
        raise ValueError(f"preprocess takes at most {CLIP_LEN} frames, got {kp.shape[1]}")
    x_max, x_min, y_max, y_min = kp[..., 0].max(), kp[..., 0].min(), kp[..., 1].max(), kp[..., 1].min()
    if (x_max - x_min) > 10 and (y_max - y_min) > 10:
        kp[..., 0] = (kp[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
        kp[..., 1] = (kp[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
    inds = np.mod(np.arange(CLIP_LEN), kp.shape[1])
    return kp[:, inds][None]  # nc=1, M, T, V, C

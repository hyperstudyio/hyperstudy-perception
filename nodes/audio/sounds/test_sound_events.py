"""sound_events: the sounds layer from the models' scores, as the bake-off's annotator built it, and the chunking of
long soundtracks."""

import numpy as np
import pytest

from sound_events import (AGREEING_TAGS, IGNORED_TAGS, PANNS_CONTEXT_S, PANNS_CORE_S, PANNS_GRID, PANNS_MPS_FRAMES, PE_CONTEXT_S, PE_CORE_S, PE_HOP,
                          QUERIES, TAG_THRESHOLD, TOP_K, Chunk, audioset_labels, padded_for_windows, plan_chunks, sound_events,
                          spans, stitch, step_max, tag_rows, tag_steps, tagger_events, windows)


def row(start, *labels):
    return {"start": float(start), "end": float(start + 1), "labels": [list(x) for x in labels]}


# --- the generic queries ---------------------------------------------------------------------------------------------

def test_every_query_names_audioset_labels_that_exist_and_speech_is_not_asked():
    labels = set(audioset_labels())
    assert len(audioset_labels()) == 527
    for query, tags in AGREEING_TAGS.items():
        assert tags and tags <= labels, (query, tags - labels)
    assert "speech" not in QUERIES and len(set(QUERIES)) == len(QUERIES)
    assert {"laughter", "music", "applause", "dog barking", "gunshot", "glass breaking", "car horn", "baby crying"} <= set(QUERIES)
    assert "Honk" not in AGREEING_TAGS["car horn"]   # AudioSet's Honk is a goose


# --- taggers ---------------------------------------------------------------------------------------------------------

def test_a_step_keeps_its_top_five_labels_that_reach_the_threshold_rounded():
    labels = [f"l{i}" for i in range(8)]
    scores = np.array([[0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.25, 0.2004]], np.float32)
    [r] = tag_rows(scores, labels)
    assert r == {"start": 0.0, "end": 1.0, "labels": [["l0", 0.9], ["l2", 0.8], ["l3", 0.7], ["l4", 0.6], ["l5", 0.5]]}
    scores = np.array([[0.9, 0.1, 0.19, 0.0, 0.0, 0.0, 0.0, 0.2]], np.float32)
    assert tag_rows(scores, labels)[0]["labels"] == [["l0", 0.9], ["l7", 0.2]]   # under the threshold: dropped
    assert TOP_K == 5 and TAG_THRESHOLD == 0.2


def test_consecutive_steps_of_a_label_are_one_event_with_its_best_score_and_speech_is_left_out():
    steps = {"panns": [row(0, ("Music", 0.5), ("Speech", 0.9)), row(1, ("Music", 0.7)), row(2, ("Speech", 0.9)),
                       row(3, ("Music", 0.3))]}
    events = tagger_events(steps)
    assert events == [{"label": "Music", "start": 0.0, "end": 2.0, "score": 0.7, "sources": ["panns"]},
                      {"label": "Music", "start": 3.0, "end": 4.0, "score": 0.3, "sources": ["panns"]}]
    assert not any(e["label"] in IGNORED_TAGS for e in events)


def test_the_two_taggers_events_of_a_label_merge_where_they_overlap_or_touch():
    steps = {"panns": [row(0, ("Music", 0.5)), row(1, ("Music", 0.6)), row(2), row(3), row(4, ("Laughter", 0.4))],
             "mn40": [row(0), row(1, ("Music", 0.8)), row(2, ("Music", 0.9)), row(3), row(4)]}
    assert tagger_events(steps) == [
        {"label": "Laughter", "start": 4.0, "end": 5.0, "score": 0.4, "sources": ["panns"]},
        {"label": "Music", "start": 0.0, "end": 3.0, "score": 0.9, "sources": ["mn40", "panns"]}]


def test_panns_frames_are_the_max_of_each_second_and_a_part_second_is_a_step():
    frames = np.zeros((250, 2), np.float32)
    frames[42, 0], frames[199, 1], frames[249, 0] = 0.9, 0.4, 0.3
    np.testing.assert_array_equal(step_max(frames, 3), np.array([[0.9, 0], [0, 0.4], [0.3, 0]], np.float32))


def test_panns_frame_at_the_very_end_of_a_whole_second_soundtrack_is_in_its_last_step_not_a_step_past_it():
    # Cnn14's STFT is centred: 2 s at 32 kHz gives 64000 // 320 + 1 = 201 frames, the last one at 2.00 s
    frames = np.zeros((201, 1), np.float32)
    frames[200, 0] = 0.5
    assert tag_steps(64000, 32000) == 2
    np.testing.assert_array_equal(step_max(frames, 2), np.array([[0], [0.5]], np.float32))


def test_each_steps_efficientat_window_is_centred_on_it_edges_padded_with_silence():
    sr = 10   # 10 samples a second: a 5 s window is 50 samples, centred on each 10-sample step
    wave = np.arange(1, 26, dtype=np.float32)   # 2.5 s: three steps
    assert tag_steps(len(wave), sr) == 3
    padded = padded_for_windows(wave, sr)
    w = windows(padded, sr, 0, 3)
    assert w.shape == (3, 50)
    np.testing.assert_array_equal(w[0, :20], 0)            # 2 s of silence before the first step
    np.testing.assert_array_equal(w[0, 20:45], wave)        # the step itself starts the clip
    np.testing.assert_array_equal(w[1, 10:35], wave)        # one step later, the window is 1 s later
    np.testing.assert_array_equal(windows(padded, sr, 1, 3), w[1:])   # a batch is the same windows


# --- queries ---------------------------------------------------------------------------------------------------------

def test_frames_above_the_threshold_are_spans_in_seconds():
    p = np.array([0.1, 0.5, 0.6, 0.2, 0.3, 0.31, 0.9], np.float32)
    assert spans(p, 48000) == [[0.04, 0.12], [0.2, 0.28]]   # 0.3 itself is not above; the last run ends with the audio
    assert spans(np.zeros(5, np.float32), 48000) == []


def test_a_query_event_agrees_when_a_tagger_event_of_a_matching_label_overlaps_it():
    steps = {"panns": [row(0, ("Laughter", 0.5)), row(1), row(2, ("Music", 0.4))]}
    events = sound_events(steps, {"laughter": [[0.5, 0.8], [1.2, 1.5]], "music": [[0.0, 0.3]]})
    assert [(e["id"], e["label"], e["start"], e["agreement"], e["sources"], e["score"]) for e in events] == [
        ("sound:0", "Laughter", 0.0, None, ["panns"], 0.5),
        ("sound:1", "music", 0.0, False, ["pe-a-frame"], None),      # Music is tagged at 2-3 s, not here
        ("sound:2", "laughter", 0.5, True, ["pe-a-frame"], None),
        ("sound:3", "laughter", 1.2, False, ["pe-a-frame"], None),   # touching is not overlapping
        ("sound:4", "Music", 2.0, None, ["panns"], 0.4)]
    assert list(events[0]) == ["id", "label", "start", "end", "sources", "agreement", "score"]   # the layer's order


# --- chunks ----------------------------------------------------------------------------------------------------------

def test_audio_no_longer_than_one_chunk_is_one_piece_whole():
    assert plan_chunks(120, 100, 10) == [Chunk(0, 120, 0, 120)]
    assert plan_chunks(50, 100, 10) == [Chunk(0, 50, 0, 50)]


def test_longer_audio_is_cores_with_context_on_both_sides_where_there_is_audio():
    assert plan_chunks(250, 100, 10) == [Chunk(0, 110, 0, 100), Chunk(90, 210, 100, 200), Chunk(190, 250, 200, 250)]


@pytest.mark.parametrize("n", [121, 250, 300, 333])
def test_stitched_chunks_give_each_frame_once_from_the_chunk_whose_core_holds_it(n):
    hop = 2
    truth = np.arange(n // hop + 1)   # a "model" whose frame i is at sample i * hop, plus one for the remainder

    def model(c):   # a frame-wise output of the chunk's samples: frame j at sample c.start + j * hop
        return np.arange(c.start // hop, c.start // hop + (c.end - c.start) // hop + 1)
    chunks = plan_chunks(n, 100, 10)
    got = stitch([(c, model(c)) for c in chunks], hop)
    np.testing.assert_array_equal(got, truth)
    two_d = stitch([(c, np.stack([model(c), -model(c)])) for c in chunks], hop, axis=1)   # (queries, frames)
    np.testing.assert_array_equal(two_d, np.stack([truth, -truth]))


def test_the_nodes_chunks_fall_on_each_models_frames_and_the_bake_offs_clips_are_one_piece():
    assert round(PANNS_CORE_S * 32000) % PANNS_GRID == 0 and round(PANNS_CONTEXT_S * 32000) % PANNS_GRID == 0
    assert PANNS_GRID == 32 * 320   # Cnn14 pools time 32-fold; a frame is 320 samples at 32 kHz
    assert round((PANNS_CORE_S + 2 * PANNS_CONTEXT_S) * 32000) // 320 + 1 < PANNS_MPS_FRAMES   # the Apple GPU's limit
    assert round(PE_CORE_S * 48000) % PE_HOP == 0 and round(PE_CONTEXT_S * 48000) % PE_HOP == 0
    for rate, core, context in ((32000, PANNS_CORE_S, PANNS_CONTEXT_S), (48000, PE_CORE_S, PE_CONTEXT_S)):
        assert len(plan_chunks(round(137.6 * rate), round(core * rate), round(context * rate))) == 1


def test_a_chunk_off_the_frame_grid_is_refused():
    with pytest.raises(ValueError, match="not on 4-sample frames"):
        stitch([(Chunk(0, 110, 0, 102), np.zeros(28)), (Chunk(92, 150, 102, 150), np.zeros(15))], 4)

"""The speakers layer from per-frame scores (speaker_links.py), without the model. Run:
uv run --project envs/gaze --with pytest pytest nodes/vision/active-speaker"""

from itertools import permutations

import numpy as np
import pytest

from speaker_links import (candidates, layer_content, link_voices, mean_score, require_diarized, speaking_intervals,
                           utterance_row, votes_for)


def best_assignment(cost):
    """The minimum-cost one-to-one assignment by trying every one (what scipy's linear_sum_assignment solves), for the
    small matrices here, which have a single best."""
    n_rows, n_cols = cost.shape
    if n_rows > n_cols:
        cols, rows = best_assignment(cost.T)
        order = np.argsort(rows)
        return rows[order], cols[order]
    best = min(permutations(range(n_cols), n_rows), key=lambda cols: sum(cost[i, j] for i, j in enumerate(cols)))
    return np.arange(n_rows), np.array(best)


def scored(index, scores, start=0.0, person=None, shot="shot:0"):
    return {"index": index, "face": f"face:{index}", "person": person, "shot": shot, "start": start, "fps": 25, "scores": scores}


def word(start, end, voice):
    return {"w": "x", "start": start, "end": end, "voice": voice}


def test_a_tracks_mean_score_runs_over_the_frames_from_starts_to_ends():
    t = scored(0, [1.0, 2.0, 3.0, 4.0, 5.0], start=1.0)
    assert mean_score(t, 1.0, 1.08) == 2.0               # frames 0 .. 2
    assert mean_score(t, 0.5, 1.0) == 1.0                # from before the track: its first frame
    assert mean_score(t, 1.04, 9.0) == 3.5               # to after it: its last
    assert mean_score(t, 0.0, 0.9) is None and mean_score(t, 2.0, 3.0) is None


def test_candidates_are_the_scored_tracks_of_the_utterances_shot_highest_first():
    tracks = [scored(0, [0.5] * 50), scored(1, [2.0] * 50), scored(2, [9.0] * 50, shot="shot:1"), scored(3, [0.5] * 50),
              scored(4, [1.0] * 10, start=5.0)]
    u = {"id": "utt:0", "start": 0.0, "end": 1.0, "shot": "shot:0"}
    assert candidates(tracks, u) == [(2.0, 1), (0.5, 3), (0.5, 0)]   # ties: the later track first; 4 is not on screen then


def test_words_vote_with_their_voice_unless_too_long_or_voiceless():
    spans = speaking_intervals([word(0, 1, "voice:A"), word(1, 3.5, "voice:A"), word(2, 2.5, None), word(3, 3.2, "voice:B")])
    assert spans == {"voice:A": [(0, 1)], "voice:B": [(3, 3.2)]}


def test_each_word_votes_for_the_visible_person_scored_highest_above_zero():
    tracks = [scored(0, [1.0] * 25 + [-1.0] * 25, person="person:P"), scored(1, [0.5] * 50, person="person:Q"),
              scored(2, [5.0] * 50, person=None),                      # no person: neither votes nor counts as seen
              scored(3, [-2.0] * 25, start=2.0, person="person:R")]
    spans = {"voice:A": [(0.1, 0.5), (1.2, 1.6)], "voice:B": [(2.1, 2.5), (5.0, 5.5)]}
    votes, visible = votes_for(spans, tracks)
    assert votes == {"voice:A": {"person:P": 1, "person:Q": 1}, "voice:B": {}}   # R scored below 0 at 2.1 s
    assert visible == {"voice:A": 2, "voice:B": 1}                               # no face at 5 s


def test_voices_and_people_are_matched_one_to_one_on_word_share():
    p, q = "person:P", "person:Q"
    tracks = [scored(0, [1.0] * 100, person=p), scored(1, [-1.0] * 50 + [2.0] * 50, person=q)]
    # A speaks 3 words while P leads and 1 while Q does; B 2 words, both while Q leads: A -> P (0.75), B -> Q (1.0)
    spans = {"voice:A": [(0.1, 0.2), (0.5, 0.6), (1.0, 1.1), (3.0, 3.1)], "voice:B": [(2.5, 2.6), (3.5, 3.6)]}
    links = link_voices(spans, tracks, [p, q], best_assignment)
    assert links == {"voice:A": {"person": p, "word_share": 0.75}, "voice:B": {"person": q, "word_share": 1.0}}
    # a third voice only Q would explain loses Q to B, and is linked to no one (share 0)
    spans["voice:C"] = [(2.1, 2.2)]
    assert "voice:C" not in link_voices(spans, tracks, [p, q], best_assignment)
    assert link_voices({}, tracks, [p, q], best_assignment) == {}


def row(ranked_scores, voice="voice:A", links=None, min_share=0.3, person_of=None):
    tracks = [scored(i, [s] * 50) for i, s in enumerate(ranked_scores)]
    return utterance_row({"id": "utt:7", "start": 0.0, "end": 1.0, "shot": "shot:0", "voice": voice}, tracks,
                         person_of or {}, links or {}, min_share)


def test_an_utterance_with_a_positive_top_face_is_that_face_and_its_person():
    r = row([0.5, 2.0], person_of={1: "person:Q"})
    assert r == {"utt": "utt:7", "predicted": {"kind": "face", "face": "face:1", "via": "asd", "person": "person:Q"},
                 "candidates": [{"face": "face:1", "score": 2.0}, {"face": "face:0", "score": 0.5}], "margin": 1.5,
                 "second_opinion_agrees": None}
    assert row([0.25])["predicted"] == {"kind": "face", "face": "face:0", "via": "asd"}   # a face with no person
    assert row([0.25])["margin"] == 0.25                                              # against nobody: 0


def test_without_a_speaking_face_a_trusted_voice_link_names_the_person():
    links = {"voice:A": {"person": "person:P", "word_share": 0.3}, "voice:B": {"person": "person:Q", "word_share": 0.29}}
    r = row([-0.5, -1.0], links=links)
    assert r["predicted"] == {"kind": "person", "person": "person:P", "voice": "voice:A", "via": "voice_link"}
    assert r["margin"] == 0.5                                                         # how far the top face is below 0
    assert row([-0.5], voice="voice:B", links=links)["predicted"] == {"kind": "offscreen", "voice": "voice:B", "via": "none"}
    assert row([], voice=None, links=links) == {"utt": "utt:7", "predicted": {"kind": "offscreen", "voice": None, "via": "none"},
                                                "candidates": [], "margin": None, "second_opinion_agrees": None}
    assert row([-0.5], links=links, min_share=0.5)["predicted"]["kind"] == "offscreen"


def test_the_layer_lists_every_utterance_and_the_voice_links_with_their_trust():
    p, q = "person:P", "person:Q"
    faces = {"people": [{"id": p}, {"id": q}], "tracks": [{"person": p}, {"person": q}]}
    tracks = [scored(0, [1.0] * 100, person=p), scored(1, [-1.0] * 75 + [2.0] * 25, person=q)]
    transcript = {"words": [word(0.1, 0.2, "voice:A"), word(3.1, 3.2, "voice:B"), word(1.0, 1.1, "voice:B"),
                            word(2.0, 2.1, "voice:B")],
                  "utterances": [{"id": "utt:0", "start": 0.1, "end": 0.2, "shot": "shot:0", "voice": "voice:A"},
                                 {"id": "utt:1", "start": 3.1, "end": 3.2, "shot": "shot:9", "voice": "voice:B"}]}
    content = layer_content(transcript, faces, tracks, 0.4, best_assignment)
    assert content["voice_links"] == [{"voice": "voice:A", "person": p, "word_share": 1.0, "trusted": True},
                                      {"voice": "voice:B", "person": q, "word_share": 0.33, "trusted": False}]
    assert [r["predicted"]["kind"] for r in content["utterances"]] == ["face", "offscreen"]   # B's link is under 0.4


def test_a_diarized_transcript_without_any_voice_is_read_and_asrs_is_refused():
    """diarize leaves every voice empty when pyannote finds no turns (its test_no_turns_leave_every_voice_empty): that
    is a valid input, whose utterances are off-screen voices. asr's transcript, which never had voices, is refused."""
    words = [word(0.1, 0.2, None), word(0.3, 0.4, None)]
    silent = {"producer": {"stage": "transcribe+diarize"}, "voices": [], "words": words,
              "utterances": [{"id": "utt:0", "start": 0.1, "end": 0.4, "shot": "shot:0", "voice": None}]}
    require_diarized(silent)
    faces = {"people": [{"id": "person:P"}], "tracks": [{"person": "person:P"}]}
    content = layer_content(silent, faces, [scored(0, [-1.0] * 25, person="person:P")], 0.3, best_assignment)
    assert content["voice_links"] == [] and content["utterances"][0]["predicted"] == {"kind": "offscreen", "voice": None, "via": "none"}
    with pytest.raises(ValueError, match="stage 'transcribe', not a diarization"):
        require_diarized({**silent, "producer": {"stage": "transcribe"}})


@pytest.mark.parametrize("share,trusted", [(0.3, True), (0.29, False)])
def test_a_link_is_trusted_at_the_gate_or_above(share, trusted):
    faces = {"people": [{"id": "person:P"}], "tracks": [{"person": "person:P"}]}
    votes = round(share * 100)
    tracks = [scored(0, [1.0] * votes + [-1.0] * (100 - votes), person="person:P")]
    words = [word(i / 25 + 0.001, i / 25 + 0.002, "voice:A") for i in range(100)]   # one word per frame
    content = layer_content({"words": words, "utterances": []}, faces, tracks, 0.3, best_assignment)
    assert content["voice_links"] == [{"voice": "voice:A", "person": "person:P", "word_share": share, "trusted": trusted}]

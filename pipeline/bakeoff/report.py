"""Stage 3 of the bake-off: one self-contained HTML page comparing the models.

One row per shot: a thumbnail of each keyframe, the speech heard during the
shot, then each model's caption side by side. A summary table at the top shows
tokens, cost, latency and errors per model, extrapolated to one video-hour.

Usage: uv run python report.py runs/<name>
"""

import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image

from caption import shot_speech

THUMB_WIDTH = 240


def thumb(path):
    image = Image.open(path)
    image.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def caption_html(shot):
    if shot["error"]:
        return f'<p class="err">{html.escape(shot["error"])}</p>'
    c = shot["caption"]
    people = "".join(
        f'<li><b>{html.escape(p["label"])}</b>{" (speaking)" if p["speaking"] else ""}: '
        f'{html.escape(p["appearance"])}; {html.escape(p["action"])}</li>' for p in c["people"])
    text = ", ".join(f"“{html.escape(t)}”" for t in c["onscreen_text"])
    return (f'<p>{html.escape(c["description"])}</p>'
            f'<p class="meta">{html.escape(c["setting"])} · {html.escape(c["mood"])}</p>'
            + (f"<ul>{people}</ul>" if people else "")
            + (f'<p class="meta">On screen: {text}</p>' if text else "")
            + f'<p class="meta">{shot["seconds"]}s · {shot["input_tokens"]}/{shot["output_tokens"]} tok</p>')


def main():
    run_dir = Path(sys.argv[1])
    extract = json.loads((run_dir / "extract.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted((run_dir / "captions").glob("*.json"))]
    hours = extract["video"]["duration"] / 3600

    summary = "".join(
        f"<tr><td>{html.escape(r['model'])}</td><td>{r['totals']['input_tokens']:,}</td>"
        f"<td>{r['totals']['output_tokens']:,}</td>"
        f"<td>{'—' if r['totals']['cost_usd'] is None else '$%.3f' % r['totals']['cost_usd']}</td>"
        f"<td>{'—' if r['totals']['cost_usd'] is None else '$%.2f' % (r['totals']['cost_usd'] / hours)}</td>"
        f"<td>{r['totals']['seconds'] / len(r['shots']):.1f}s</td><td>{r['totals']['errors']}</td></tr>"
        for r in results)

    faces_html = ""
    faces_path = run_dir / "faces.json"
    if faces_path.exists():
        faces = json.loads(faces_path.read_text())
        cards = "".join(
            f'<figure><img src="{thumb(run_dir / p["crop"])}" alt="{html.escape(p["label"])}">'
            f'<figcaption>{html.escape(p["label"])}<br>shots {", ".join(map(str, p["shots"]))}<br>'
            f'{p["seconds_on_screen"]}s on screen</figcaption></figure>' for p in faces["people"])
        faces_html = (f'<h2>Face identities (py-feat Detectorv2, {faces["sample_fps"]} fps)</h2>'
                      f'<p class="sub">{faces["detections"]} detections → {len(faces["tracks"])} tracks → '
                      f'{len(faces["people"])} people. {html.escape(faces["licence"])}</p><div class="faces">{cards}</div>')

    social_path, links_path = run_dir / "social.json", run_dir / "speaker_links.json"
    social = json.loads(social_path.read_text()) if social_path.exists() else None
    if links_path.exists():
        links = json.loads(links_path.read_text())["links"]
        items = "".join(f"<li>{html.escape(s)} → {html.escape(l['person'] or 'not on screen')}"
                        + (f" (mouth Δ {l['mouth_diff']})" if l.get("mouth_diff") is not None else "") + "</li>"
                        for s, l in sorted(links.items()))
        faces_html += (f"<h2>Voices → faces</h2><p class=\"sub\">py-feat AU25/AU26 mouth movement while each "
                       f"diarized voice talks vs. not. Unverified except where noted in the summary.</p><ul>{items}</ul>")

    head = ("<th>Face signals (py-feat, unvalidated)</th>" if social else "") + "".join(
        f"<th>{html.escape(r['model'])}</th>" for r in results)
    rows = []
    for shot in extract["shots"]:
        frames = "".join(f'<img src="{thumb(run_dir / k["file"])}" alt="frame at {k["t"]}s">' for k in shot["keyframes"])
        speech = html.escape(shot_speech(shot, extract["transcript"])).replace("\n", "<br>")
        cells = "".join(f"<td>{caption_html(r['shots'][shot['index']])}</td>" for r in results)
        if social:
            people = social["shots"][shot["index"]]["people"]
            signals = "".join(
                f"<li><b>{html.escape(name)}</b>: looks at {html.escape(p['target'] or 'unclear')}"
                + (" (mutual)" if p["mutual_gaze"] else "")
                + (f"; {html.escape(p['expression'].lower())} (+{p['expression_rise']:.2f} vs own baseline)" if p["expression"] else "")
                + "</li>" for name, p in sorted(people.items()))
            cells = f'<td class="signals">{"<ul>" + signals + "</ul>" if signals else "—"}</td>' + cells
        rows.append(f'<tr><td class="shot"><b>Shot {shot["index"]}</b> · {shot["start"]:.1f}–{shot["end"]:.1f}s'
                    f'<div class="frames">{frames}</div><p class="speech">{speech}</p></td>{cells}</tr>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caption Bake-off</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e3e3e6; --err:#b42318; --card:#f6f6f8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --fg:#ececf0; --muted:#9a9aa3; --line:#2c2c31; --err:#ff8a80; --card:#1d1d21; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; padding:24px 16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin:0 0 20px; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ border-top:1px solid var(--line); padding:10px; vertical-align:top; text-align:left; }}
th {{ position:sticky; top:0; background:var(--bg); }}
.summary td, .summary th {{ padding:6px 10px; }} .summary {{ width:auto; margin-bottom:28px; }}
td.shot {{ width:260px; background:var(--card); }} .frames img {{ width:240px; display:block; margin:6px 0; border-radius:4px; }}
.speech {{ font-style:italic; color:var(--muted); }} .meta {{ color:var(--muted); font-size:12px; margin:4px 0; }}
ul {{ margin:4px 0; padding-left:18px; }} p {{ margin:0 0 6px; }} .err {{ color:var(--err); }}
td.signals {{ width:200px; font-size:12px; }} .wrap {{ overflow-x:auto; }} h2 {{ font-size:16px; margin:28px 0 6px; }}
.faces {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px; }} .faces figure {{ margin:0; width:120px; font-size:12px; color:var(--muted); }}
.faces img {{ width:120px; height:120px; object-fit:cover; border-radius:4px; display:block; }}
</style></head><body>
<h1>Caption bake-off: {html.escape(Path(extract['video']['source']).name)}</h1>
<p class="sub">{extract['video']['duration']:.1f}s · {len(extract['shots'])} shots · transcript: WhisperX {html.escape(extract['transcript']['model'])},
diarized={extract['transcript']['diarized']} · sha256 {extract['video']['sha256'][:12]}</p>
<table class="summary"><tr><th>Model</th><th>Input tok</th><th>Output tok</th><th>Cost (this clip)</th><th>Cost per video-hour</th><th>Avg s/shot</th><th>Errors</th></tr>{summary}</table>
{faces_html}
<h2>Per-shot captions</h2>
<div class="wrap"><table><tr><th>Shot</th>{head}</tr>{''.join(rows)}</table></div>
</body></html>"""
    out = run_dir / "bakeoff.html"
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()

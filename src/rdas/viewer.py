"""Build the self-contained replay viewer from exported runs.

    python -m rdas.viewer results/live/eft-retry/run.json results/live/adaptive/run.json -o results/viewer.html

The output is one HTML file with the run data embedded: no server, no internet,
opens in any browser. Space = play / pause, arrow keys = step between events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def build_viewer(runs: list[dict], out_path, title: str = "", subtitle: str = "", note: str = "") -> Path:
    if not runs:
        raise ValueError("need at least one run")
    data = {"runs": runs, "title": title, "subtitle": subtitle, "note": note}
    # Embedded in <script type="application/json">: neutralise "</" so no value can close the tag.
    payload = json.dumps(data).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__RDAS_DATA__", payload)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the replay viewer from run.json files.")
    ap.add_argument("runs", nargs="+", help="run.json files (from rdas.live or scripts/record_demo.py)")
    ap.add_argument("-o", "--out", default="results/viewer.html")
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    a = ap.parse_args(argv)
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.runs]
    out = build_viewer(runs, a.out, a.title, a.subtitle)
    print(f"viewer written to {out.resolve()}")


if __name__ == "__main__":
    main()

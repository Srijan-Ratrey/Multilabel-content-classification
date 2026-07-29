"""Shared probability/threshold chart, used by both the Gradio and Streamlit UIs.

Kept out of app.py so the Streamlit deployment does not need gradio as a dependency.

Form: horizontal bars (magnitude) with a per-bar threshold tick, coloured by STATE
(flagged / below threshold) rather than by six categorical hues, because state is what the
chart is about. State is additionally carried by the label name, the numeric value and
explicit FLAGGED/below text, so colour is never the sole encoding.
"""

from __future__ import annotations

import html as html_lib

import numpy as np

from src.config import LABELS


# Colours are taken from the data-viz reference palette rather than invented, since a
# JS runtime was unavailable to run its validator. `critical` is a fixed status step
# (documented contrast 4.68 on the light surface, 3.62 on dark); the neutral is the
# secondary ink token (7.73 / 9.72). Both clear the 3:1 mark-vs-surface floor in both
# modes, verified numerically. State is additionally carried by the label name, the
# numeric value, and explicit "FLAGGED"/"below" text, so colour is never alone.
CSS = """
.viz-root {
  --surface-1: #fcfcfb; --surface-2: #f2f2ef;
  --text-primary: #0b0b0b; --text-secondary: #52514e;
  --mark-flagged: #d03b3b; --mark-below: #52514e; --rule: #d9d8d2;
  font-variant-numeric: tabular-nums;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1: #1a1a19; --surface-2: #24241f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --mark-flagged: #d03b3b; --mark-below: #c3c2b7; --rule: #3a3a35;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1: #1a1a19; --surface-2: #24241f;
  --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --mark-flagged: #d03b3b; --mark-below: #c3c2b7; --rule: #3a3a35;
}
.viz-root { background: var(--surface-1); padding: 14px 16px; border-radius: 10px; }
.viz-legend { display: flex; gap: 18px; align-items: center; margin-bottom: 12px;
  font-size: 12px; color: var(--text-secondary); flex-wrap: wrap; }
.viz-key { display: inline-flex; gap: 6px; align-items: center; }
.viz-swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.viz-row { display: grid; grid-template-columns: 112px 1fr 108px; gap: 10px;
  align-items: center; margin-bottom: 7px; }
.viz-name { font-size: 13px; color: var(--text-primary); text-align: right; }
/* the track is the 0..1 domain; 2px inset keeps a surface gap around the fill */
.viz-track { position: relative; height: 20px; background: var(--surface-2);
  border-radius: 4px; overflow: visible; }
.viz-fill { position: absolute; left: 0; top: 2px; bottom: 2px;
  border-radius: 0 4px 4px 0; }          /* 4px rounded data-end, anchored at the baseline */
.viz-thr { position: absolute; top: -3px; bottom: -3px; width: 2px;
  background: var(--text-secondary); }
.viz-thr::after { content: attr(data-t); position: absolute; top: -14px; left: -10px;
  font-size: 10px; color: var(--text-secondary); white-space: nowrap; }
.viz-val { font-size: 12px; color: var(--text-secondary); }
.viz-val b { color: var(--text-primary); font-size: 13px; }
.viz-verdict { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--text-primary); }
.viz-note { font-size: 11.5px; color: var(--text-secondary); margin-top: 6px; }
"""


def bar_chart(probs: np.ndarray, thresholds: dict[str, float], mode_label: str) -> str:
    rows = []
    for i, label in enumerate(LABELS):
        p, t = float(probs[i]), float(thresholds[label])
        fired = p >= t
        colour = "var(--mark-flagged)" if fired else "var(--mark-below)"
        state = "FLAGGED" if fired else "below"
        rows.append(
            f'<div class="viz-row">'
            f'<div class="viz-name">{label}</div>'
            f'<div class="viz-track">'
            f'  <div class="viz-fill" style="width:{max(p, 0.004) * 100:.2f}%;background:{colour};"></div>'
            f'  <div class="viz-thr" style="left:{t * 100:.2f}%;" data-t="{t:.2f}"></div>'
            f"</div>"
            f'<div class="viz-val"><b>{p:.3f}</b> · {state}</div>'
            f"</div>"
        )

    fired = [l for i, l in enumerate(LABELS) if probs[i] >= thresholds[l]]
    verdict = ", ".join(fired) if fired else "no labels"
    return (
        f'<div class="viz-root">'
        f'<div class="viz-legend">'
        f'  <span class="viz-key"><span class="viz-swatch" style="background:var(--mark-flagged)"></span>'
        f"    at or above threshold (flagged)</span>"
        f'  <span class="viz-key"><span class="viz-swatch" style="background:var(--mark-below)"></span>'
        f"    below threshold</span>"
        f'  <span class="viz-key">| vertical tick = that label\'s threshold</span>'
        f"</div>"
        + "".join(rows)
        + f'<div class="viz-verdict"><b>Predicted:</b> {html_lib.escape(verdict)} '
        f'<span class="viz-note">({mode_label})</span></div>'
        f'<div class="viz-note">Probabilities sum to {probs.sum():.2f}. They are six independent '
        f"sigmoids, not a distribution — nothing constrains them to 1.0.</div>"
        f"</div>"
    )



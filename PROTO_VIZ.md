# Prototype Visualizer — `proto_viz_app.py`

Web GUI for inspecting how MultiProtoPTA builds and uses patch-level prototype
banks during test-time adaptation. Runs in two modes: **live** (loads CLIP and
a dataset, processes samples in real time) or **replay** (loads pre-saved
records from a previous live run).

---

## Requirements

- All packages from the `pta` conda environment
- Web dependencies in `requirements.txt`: `fastapi`, `uvicorn`, `jinja2`
- Browser access to the host/port where the server runs

---

## Quick start

```bash
conda activate /share_98/projects/brandon/envs/pta

# Start web app (default: http://127.0.0.1:8000)
python proto_viz_app.py

# Expose on all interfaces
python proto_viz_app.py --host 0.0.0.0 --port 8000

# Preload a replay file at startup
python proto_viz_app.py --records outputs/viz/caltech101_n500.pkl
```

---

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Host interface for uvicorn. |
| `--port` | `8000` | Port for uvicorn. |
| `--reload` | off | Enable auto-reload for web development. |
| `--records PATH` | `None` | Optionally preload pre-saved `.pkl` records on startup. |

---

## Web GUI controls

```
Top controls include:
- `Next Sample` button (advance by one sample)
- `Restart` button (jump back to sample 0)
- `Load` controls for choosing dataset/mode and live settings
- `Save Current` button (exports current sample info as JSON)
```

The page still shows the same score breakdown table, top prototype match table,
and per-class bank stats used in the Tkinter tool.

**Score breakdown table** — one row per class, sorted by final logit
(highest = most likely prediction). Columns:

| Column | Meaning |
|---|---|
| `class` | Class name |
| `bank_K` | Number of patch prototypes currently stored for this class |
| `text_score` | `100 × cos(text_embedding, global_image_feature)` — the CLIP zero-shot contribution |
| `raw_proto` | Evidence-weighted prototype bank score before mean-centering |
| `centered` | `raw_proto − mean(raw_proto over all classes)` — what actually gets added to the logit |
| `final_logit` | `text_score + τ × centered` — the number argmax is taken over |
| `prob%` | `softmax(final_logits)[c]` as a percentage |

Rows are highlighted: **green** = ground-truth class, **red** = predicted class
when the prediction is wrong.

**Top prototype matches table** — the 20 highest cosine-similarity hits between
any patch of the current image and any stored prototype across all classes.
Useful for seeing which prototypes are activating and whether they belong to the
correct class.

| Column | Meaning |
|---|---|
| `class` | Class that owns this prototype |
| `proto#` | Index of the prototype within that class's bank |
| `appear` | Number of images this prototype has been matched in |
| `evid_w` | Evidence weight: `min(1, 0.5 + 0.5×(K−1)/4)` — ramps from 0.5 at K=1 to 1.0 at K=5 |
| `max_sim` | Highest cosine similarity between any patch in the current image and this prototype |
| `centered_s` | Per-prototype contribution to the centered score |
| `contrib` | `τ × centered_logit[c]` — actual delta added to the final logit for this class |

**Bank stats panel** — detailed breakdown for the class selected in the combobox
(defaults to the ground-truth class). Click any row in the score breakdown table
to switch the selected class.

---

## Saved records and exports

Live mode saves a `.pkl` file to `outputs/viz/<dataset>_n<N>.pkl`. Each record
is a `SampleRecord` (defined in `utils/proto_viz.py`) containing:

- The global image feature and all patch embeddings `[P, D]`
- Per-class score breakdown (`ClassRecord` list)
- Top-20 prototype matches (`MatchInfo` list), each with the top-3 patch indices
  that produced the match
- Running accuracy up to that sample

Records are plain Python dataclasses serialised with `pickle`, so you can load
and inspect them in a notebook:

```python
from utils.proto_viz import ProtoVizEngine, SampleRecord

records = ProtoVizEngine.load_records("outputs/viz/caltech101_n500.pkl")

rec = records[42]
print(rec.target_name, "→", rec.predicted_name, "correct:", rec.correct)

# Top match
m = rec.top_matches[0]
print(m.proto.class_name, "proto", m.proto.proto_idx,
      "sim=", m.max_patch_sim, "appear=", m.proto.appearance)

# All class scores
for cr in sorted(rec.classes, key=lambda c: -c.final_logit)[:5]:
    print(f"  {cr.class_name:<30} logit={cr.final_logit:.2f}  K={cr.bank_size}")
```

`Save Current` from the web UI writes one JSON file to `outputs/viz_exports/`
using:

`YYYYmmdd_HHMMSS_<dataset>_sample<idx>.json`

Example:

`20260609_143512_caltech101_sample00042.json`

---

## Reusable utils layer

Frontend-independent helpers now live in `utils/proto_viz_session.py`:

- `list_available_datasets(config_dir)`
- `sample_record_to_dict(record, selected_class_name=None)`
- `ProtoVizSession.load_live(...)`
- `ProtoVizSession.load_replay(...)`
- `ProtoVizSession.next()` / `restart()` / `set_index(idx)`
- `ProtoVizSession.export_current(output_dir="outputs/viz_exports")`

These functions are documented with docstrings and are intended for reuse by
future debugging tools (web UI, notebook widgets, CLI inspectors, etc.).

---

## How the engine mirrors MultiProtoPTA

`ProtoVizEngine` (`utils/proto_viz.py`) is a faithful reimplementation of
`MultiProtoPTABase.run()` that records all intermediate values instead of
discarding them. It reads the same config keys with the same defaults:

| Config key | Default | Effect |
|---|---|---|
| `max_K` | 30 | Max prototypes per class |
| `match_threshold` | 0.8 | Cosine threshold for patch-to-prototype matching |
| `conf_threshold` | 0.1 | Min CLIP confidence to trigger a prototype update |
| `tau_proto` / `T` | 20.0 | Scale factor on the centered proto score |
| `proto_top_n` | 5 | Top-K prototypes averaged for the bank score |
| `disable_centering` | False | Skip mean-centering of proto scores |
| `penalize_common` | True | Apply cross-class commonness penalty |
| `common_threshold` | 0.7 | Cosine threshold for the commonness penalty |
| `topk_update` | 1 | Number of top-confidence classes updated per sample |
| `adaptive_tau` | False | Scale τ by proto-score variance |

Pass these in your YAML config alongside `alpha` and `T` to override defaults.

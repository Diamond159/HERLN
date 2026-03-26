# Project Guidelines

## Code Style
- This repo is a Python research codebase; prefer small, flag-gated changes over broad refactors.
- Add new runtime options in [src/config.py](src/config.py), then thread them through [src/main.py](src/main.py) into [model/rrgcn.py](model/rrgcn.py).
- Reuse helpers in [src/utils.py](src/utils.py) for snapshot splitting, graph building, answer filtering, checkpoint metadata, and community graph creation.
- Preserve compatibility with the pinned older stack in [requirement.txt](requirement.txt), including older DGL patterns and the `TAGConv` fallback in [model/rrgcn.py](model/rrgcn.py).
- Follow existing mixed English/Chinese comments where they already clarify research logic; do not normalize unrelated comments while editing.

## Architecture
- [src/main.py](src/main.py) is the main orchestration entrypoint for training and evaluation.
- Local temporal KG datasets are loaded from [data](data) through [src/knowledge_graph.py](src/knowledge_graph.py); each dataset directory is expected to contain `entity2id.txt`, `relation2id.txt`, `train.txt`, `valid.txt`, `test.txt`, and usually `train.csv`.
- The core model is composed in [model/rrgcn.py](model/rrgcn.py): encoder/decoder wiring, class-graph propagation, contextual relation prior, Lie regularization, and optional ERD-Net extensions are coordinated there.
- Community/class information is part of the model input, not just preprocessing: [src/main.py](src/main.py) builds a class graph from `train.csv`, and [model/rrgcn.py](model/rrgcn.py) consumes it.
- Optional ERD-Net modules live in [model/relation_dynamics.py](model/relation_dynamics.py) and [model/copy_generation_decoder.py](model/copy_generation_decoder.py); extend those seams instead of duplicating logic elsewhere.

## Build and Test
- Install dependencies from [requirement.txt](requirement.txt). The repo currently pins `torch==1.6.0`, `torchvision==0.7.0`, and `dgl-cu102==0.5.2`.
- Safe validation commands to try first from repo root:
  - `python -m py_compile src/main.py src/config.py src/utils.py src/knowledge_graph.py model/rrgcn.py model/relation_dynamics.py model/copy_generation_decoder.py`
  - `python test_simple.py`
  - `python verify_erd_integration.py`
  - `python test_erd_net_features.py`
- Follow the documented training flow in [README.md](README.md):
  - `cd src`
  - `python main.py -d ICEWS14s --self-loop --layer-norm --weight 0.5 --theta 1 --entity-prediction --relation-prediction --gpu 0`
- Prefer the current CLI in [src/config.py](src/config.py) over printed examples inside helper scripts; some helper-script training commands reference stale flags not present in the active parser.

## Project Conventions
- Temporal snapshot logic in [src/utils.py](src/utils.py) assumes records are sorted by timestamp and groups by the 4th column.
- Checkpoints are written under dated folders in [checkpoints](checkpoints) as `year/month/day/hour/run/`; preserve that layout when adding outputs.
- Root-level scripts such as [test_simple.py](test_simple.py), [verify_erd_integration.py](verify_erd_integration.py), and [test_erd_net_features.py](test_erd_net_features.py) are ad hoc validation helpers, not a formal `pytest` suite.
- Keep experimental features off by default unless the repo already enables them via parser defaults.

## Integration Points
- Relation prediction extensions already have hooks for contextual priors, global relation dynamics, and copy-generation in [model/rrgcn.py](model/rrgcn.py).
- Line-graph / PM-PD functionality is centralized in [src/utils.py](src/utils.py) and gated by `--enable-line-graph` in [src/config.py](src/config.py).
- Data loading is local-first; prefer working with the checked-in datasets under [data](data) rather than adding new download paths.

## Security
- Treat checkpoint loading in [src/main.py](src/main.py) as trusted-local only; do not introduce automatic loading of untrusted `.pt` files.
- Avoid destructive cleanup of [checkpoints](checkpoints) or [data](data); both directories are part of the working research state.

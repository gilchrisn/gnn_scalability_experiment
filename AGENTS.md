# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the production Python code: `commands/` (CLI actions), `kernels/` (exact/KMV/MPRW materialization), `bridge/` (C++/AnyBURL adapters), `data/`, `backend/`, `analysis/`, and model modules.
- `scripts/` contains experiment pipelines (`exp1_partition.py` to `exp5_degree_bin_eval.py`) and paper/extension runners.
- `tests/` contains `pytest` suites (currently focused on MPRW kernel behavior).
- `HUB/` contains C++ sources for graph materialization; `make` builds `bin/graph_prep`./co
- Runtime artifacts belong in ignored folders such as `results/`, `figures/`, `output/`, and dataset directories.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `make clean && make`: rebuild the C++ helper binary used by bridge/pipeline scripts.
- `python main.py -h`: list CLI subcommands.
- `python main.py benchmark --dataset HGB_ACM --method exact`: run a baseline benchmark path.
- `bash run_experiments.sh`: execute the full staged pipeline (partition, train, inference, visualize).
- `pytest tests/ -v`: run all tests.
- `pytest tests/test_mprw_kernel.py -v`: run kernel-focused regression tests.

## Coding Style & Naming Conventions
- Use 4-space indentation, PEP 8-friendly formatting, and import grouping (stdlib, third-party, local).
- Follow `markdown/style.md`: type hints on function signatures and Google-style docstrings for public APIs.
- Naming: `PascalCase` classes, `snake_case` functions/modules, `UPPER_SNAKE_CASE` constants.
- Keep modules focused by responsibility (e.g., new kernels under `src/kernels/`, not mixed into CLI code).

## Testing Guidelines
- Framework: `pytest`.
- Test files should be named `test_*.py`; test functions should be `test_*` and deterministic (explicit seeds).
- Add or update tests for changes to kernels, bridge execution, and dataset parsing paths.
- No enforced coverage threshold is configured; prioritize behavior/regression coverage for modified code.

## Commit & Pull Request Guidelines
- Current history includes many short placeholder commit messages; for new work, use descriptive imperative summaries (example: `scripts: add max-rss guard to exp3 inference`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, datasets/metapaths affected, exact commands run, and key output locations (for example `results/<dataset>/master_results.csv`, `figures/`).
- If plots or metrics change, include before/after evidence in the PR description.

.PHONY: env install test lint format data-help clean

env:  ## create the conda env (no GPU packages, no data)
	conda env create -f environment.yml || conda env update -f environment.yml

install:  ## editable install with dev tools (CPU only)
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q -m "not gpu and not data and not slow"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

data-help:  ## print how to obtain datasets; downloads nothing
	@echo "Datasets are never committed and never fetched by default targets."
	@echo "See data/README.md and scripts/data/ (if present) for per-dataset instructions."
	@echo "Skoltech3D (CC BY-NC 4.0), Replica (research-only), DFC2019/US3D (approval required)."
	@echo "Check each licence before downloading; see README.md licence table."

clean:
	rm -rf build dist .pytest_cache .ruff_cache *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

PYTHON ?= python
MEMCAP ?=
THREADS ?= 2

# Pin BLAS / OpenMP threads: reproducible timings and no oversubscription.
export OMP_NUM_THREADS = $(THREADS)
export OPENBLAS_NUM_THREADS = $(THREADS)
export MKL_NUM_THREADS = $(THREADS)
export VECLIB_MAXIMUM_THREADS = $(THREADS)

.PHONY: install install-locked data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Exact package versions that produced the committed reports/ (see requirements.lock).
install-locked:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

# Download the raw archives (KDD Cup 1999 10%, CoverType) into data/raw/ and cache the subsets.
data:
	$(MEMCAP) $(PYTHON) -m anomalybench.cli download --data-home data

# Full reproduction: benchmark -> reports/metrics.json + reports/figures/*.png.
run: data
	$(MEMCAP) $(PYTHON) -m anomalybench.cli run --data-home data --out reports --threads $(THREADS)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info

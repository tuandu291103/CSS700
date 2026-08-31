.PHONY: install install-research test lint format doctor pilot clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-research:
	$(PYTHON) -m pip install -e ".[research,dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

doctor:
	$(PYTHON) -m css700.cli doctor

pilot:
	$(PYTHON) -m css700.cli pilot --config configs/pilot.yaml

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

.PHONY: install install-research test lint format doctor pilot train-grid grad-cam integrated-gradients occlusion faithfulness grid-stability train-tabular-baselines audit-som-mapping clean

.PHONY: tree-shap

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-research:
	$(PYTHON) -m pip install -e ".[research,dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

doctor:
	PYTHONPATH=src $(PYTHON) -m css700.cli doctor

pilot:
	PYTHONPATH=src $(PYTHON) -m css700.cli pilot --config configs/pilot.yaml

train-grid:
	PYTHONPATH=src $(PYTHON) -m css700.cli train-grid --config configs/grid_cnn.yaml

grad-cam:
	PYTHONPATH=src $(PYTHON) -m css700.cli grad-cam --run-dir "$(RUN_DIR)"

integrated-gradients:
	PYTHONPATH=src $(PYTHON) -m css700.cli integrated-gradients --run-dir "$(RUN_DIR)"

occlusion:
	PYTHONPATH=src $(PYTHON) -m css700.cli occlusion --run-dir "$(RUN_DIR)"

faithfulness:
	PYTHONPATH=src $(PYTHON) -m css700.cli faithfulness --run-dir "$(RUN_DIR)"

grid-stability:
	PYTHONPATH=src $(PYTHON) -m css700.cli grid-stability $(foreach run,$(RUN_DIRS),--run-dir "$(run)")

train-tabular-baselines:
	PYTHONPATH=src $(PYTHON) -m css700.cli train-tabular-baselines --config configs/tabular_baselines.yaml

audit-som-mapping:
	PYTHONPATH=src $(PYTHON) -m css700.cli audit-som-mapping --config configs/som_mapping_audit.yaml

tree-shap:
	PYTHONPATH=src $(PYTHON) -m css700.cli tree-shap --config configs/tree_shap_seed_42_v1_1.yaml

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

.PHONY: install install-research test lint format doctor pilot train-grid grad-cam integrated-gradients occlusion faithfulness grid-stability train-tabular-baselines audit-som-mapping clean

.PHONY: tree-shap tree-shap-replicas tree-shap-stability lime-preflight lime-locality lime-test
.PHONY: fetch-spambase audit-spambase split-spambase split-spambase-v2
.PHONY: preflight-spambase
.PHONY: train-spambase-grid

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

tree-shap-replicas:
	PYTHONPATH=src $(PYTHON) -m css700.cli tree-shap --config configs/tree_shap_seed_7_v1_1.yaml
	PYTHONPATH=src $(PYTHON) -m css700.cli tree-shap --config configs/tree_shap_seed_2026_v1_1.yaml

tree-shap-stability:
	PYTHONPATH=src $(PYTHON) -m css700.cli tree-shap-stability $(foreach run,$(RUN_DIRS),--run-dir "$(run)")

lime-preflight:
	PYTHONPATH=src $(PYTHON) -m css700.cli lime-preflight --config configs/lime_seed_42_v1.yaml

lime-locality:
	PYTHONPATH=src $(PYTHON) -m css700.cli lime-locality --config configs/lime_locality_diagnostic_v1.yaml

lime-test:
	PYTHONPATH=src $(PYTHON) -m css700.cli lime-test --config configs/lime_seed_42_v1.yaml

fetch-spambase:
	PYTHONPATH=src $(PYTHON) -m css700.cli fetch-spambase

audit-spambase:
	PYTHONPATH=src $(PYTHON) -m css700.cli audit-spambase --source-dir "$(SOURCE_DIR)"

split-spambase:
	PYTHONPATH=src $(PYTHON) -m css700.cli split-spambase --config configs/spambase_split_v1.yaml

split-spambase-v2:
	PYTHONPATH=src $(PYTHON) -m css700.cli split-spambase-v2 --config configs/spambase_split_v2.yaml

preflight-spambase:
	PYTHONPATH=src $(PYTHON) -m css700.cli preflight-spambase --config configs/spambase_preflight_v1.yaml

train-spambase-grid:
	PYTHONPATH=src $(PYTHON) -m css700.cli train-spambase-grid --config configs/spambase_grid_seed_42_v1.yaml

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

# LASERNet Makefile
# Commands for training, testing, and running on HPC interactive nodes

.PHONY: help init clean MICROnet_notebook submit_MICROnet_notebook

# Default target: run full pipeline
all: init tempnet_notebook micronet_notebook

.DEFAULT_GOAL := all

init:
	@command -v uv >/dev/null 2>&1 || (echo "uv not found, installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync
	uv run python -m ipykernel install --user --name=.venv
	uv run nbstripout --install

# Default target: show help
help:
	@echo "LASERNet - Available Make Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  make init           - Install uv and sync dependencies"
	@echo ""
	@echo "Batch Jobs:"
	@echo "  make tempnet_notebook           - Execute notebooks/temperature-prediction.ipynb"
	@echo "  make micronet_notebook          - Execute notebooks/MICROnet.ipynb"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove logs, runs, and cache files"

# ==================== BATCH JOB SUBMISSION ====================

tempnet_notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace --debug notebooks/temperature-prediction.ipynb

micronet_notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace --debug notebooks/MICROnet.ipynb

# ==================== CLEANUP ====================

clean:
	@echo "Cleaning up logs, runs, and cache files..."
	rm -rf logs/*.out logs/*.err
	rm -rf __pycache__/
	rm -rf lasernet/__pycache__/
	rm -rf lasernet/**/__pycache__/
	rm -rf notebooks/MICROnet_output
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "Cleanup complete!"

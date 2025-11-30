# LASERNet Makefile
# Commands for training, testing, and running on HPC interactive nodes

# Environment variables
#export BLACKHOLE := /Users/bn/dtu/LASERNet/BLACKHOLE

.PHONY: help init clean MICROnet_notebook submit_MICROnet_notebook

init:
	@command -v uv >/dev/null 2>&1 || (echo "uv not found, installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync
	uv run python -m ipykernel install --user --name=.venv

# Default target: show help
help:
	@echo "LASERNet - Available Make Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  make init           - Install uv and sync dependencies"
	@echo ""
	@echo "Batch Jobs:"
	@echo "  make MICROnet_notebook          - Execute MICROnet.ipynb locally"
	@echo "  make submit_MICROnet_notebook   - Submit MICROnet notebook to job queue"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove logs, runs, and cache files"

# ==================== BATCH JOB SUBMISSION ====================

MICROnet_notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace --debug MICROnet.ipynb

submit_MICROnet_notebook:
	@echo "Submitting MICROnet_notebook to job queue"
	bsub < batch/scripts/train_MICROnet_notebook.sh

# ==================== CLEANUP ====================

clean:
	@echo "Cleaning up logs, runs, and cache files..."
	rm -rf logs/*.out logs/*.err
	rm -rf __pycache__/
	rm -rf lasernet/__pycache__/
	rm -rf lasernet/**/__pycache__/
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "Cleanup complete!"

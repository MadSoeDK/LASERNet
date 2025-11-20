# LASERNet Makefile
# Commands for training, testing, and running on HPC interactive nodes

.PHONY: help train train-micro test test-micro test-volta test-sxm2 test-a100 submit submit-micro clean

init:
	@command -v uv >/dev/null 2>&1 || (echo "uv not found, installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync

# Default target: show help
help:
	@echo "LASERNet - Available Make Commands"
	@echo "==================================="
	@echo ""
	@echo "Training (batch submission):"
	@echo "  make train          - Submit temperature prediction job to HPC"
	@echo "  make train-micro    - Submit microstructure prediction job to HPC"
	@echo ""
	@echo "Testing (local/interactive):"
	@echo "  make test           - Run quick test of temperature model (CPU)"
	@echo "  make test-micro     - Run quick test of microstructure model (CPU)"
	@echo ""
	@echo "Interactive GPU testing:"
	@echo "  make test-volta     - Test on Volta V100 GPU (voltash)"
	@echo "  make test-sxm2      - Test on Volta V100-SXM2 GPU (sxm2sh)"
	@echo "  make test-a100      - Test on A100 GPU (a100sh)"
	@echo ""
	@echo "Aliases:"
	@echo "  make submit         - Same as 'make train'"
	@echo "  make submit-micro   - Same as 'make train-micro'"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove logs, runs, and cache files"

# ==================== BATCH JOB SUBMISSION ====================

train:
	@echo "Submitting temperature prediction job to HPC..."
	bsub < batch/scripts/train.sh
	@echo "Job submitted. Check status with: bjobs"

train-micro:
	@echo "Submitting microstructure prediction job to HPC..."
	bsub < batch/scripts/train_microstructure.sh
	@echo "Job submitted. Check status with: bjobs"

submit: train

submit-micro: train-micro

# ==================== LOCAL TESTING (CPU) ====================

test:
	@echo "Running quick test of temperature model (CPU)..."
	uv run python test_microstructure.py

test-micro:
	@echo "Running quick test of microstructure model (CPU)..."
	uv run python test_microstructure.py

# ==================== INTERACTIVE GPU TESTING ====================

test-a100:
	@echo "========================================================================"
	@echo "Testing microstructure model on A100 GPU"
	@echo "========================================================================"
	@echo ""
	@echo "This will:"
	@echo "  1. Request an interactive shell on a100sh (2x A100 40GB)"
	@echo "  2. Load CUDA module and find free GPU"
	@echo "  3. Activate the virtual environment"
	@echo "  4. Run the test script"
	@echo ""
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read dummy; \
	a100sh -c "cd $(PWD) && \
		module load cuda/12.4 && \
		export CUDA_VISIBLE_DEVICES=\$$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1) && \
		echo 'Using GPU: '\$$CUDA_VISIBLE_DEVICES && \
		source .venv/bin/activate && \
		python test_microstructure.py"

# ==================== CLEANUP ====================

clean:
	@echo "Cleaning up logs, runs, and cache files..."
	rm -rf logs/*.out logs/*.err
	rm -rf runs/*/
	rm -rf runs_microstructure/*/
	rm -rf __pycache__/
	rm -rf lasernet/__pycache__/
	rm -rf lasernet/**/__pycache__/
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "Cleanup complete!"

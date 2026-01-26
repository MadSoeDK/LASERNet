"""Sequence length experiments for ConvLSTM and PredRNN baselines.

Runs experiments with seq_len = 2, 4 for both baseline models.
"""
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only run the seq_len experiments for ConvLSTM and PredRNN baselines
EXPERIMENTS = [
    "configs/experiments/seq_len/base_convlstm_micro_seq2.yaml",
    "configs/experiments/seq_len/base_convlstm_micro_seq4.yaml",
    #"configs/experiments/seq_len/base_predrnn_micro_seq2.yaml",
    #"configs/experiments/seq_len/base_predrnn_micro_seq4.yaml",
]


def build_cli_args(config: dict, extra_args: dict = None) -> list:
    """Convert config dict to CLI arguments."""
    args = []
    if extra_args:
        config = {**config, **extra_args}

    for key, value in config.items():
        cli_key = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                args.append(f"--{cli_key}")
        elif value is not None:
            args.extend([f"--{cli_key}", str(value)])

    return args


def setup_experiment_logger(config_path: Path) -> logging.Logger:
    """Create a logger with file handler for this experiment."""
    exp_name = config_path.stem
    log_dir = Path("logs/experiments")
    log_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = logging.getLogger(exp_name)
    exp_logger.setLevel(logging.INFO)

    log_file = log_dir / f"{exp_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    if not exp_logger.handlers:
        exp_logger.addHandler(file_handler)

    return exp_logger


def train_experiment(config_path: Path, wandb_group: str | None = None):
    """Run only the training step for one experiment."""
    exp_logger = setup_experiment_logger(config_path)
    exp_logger.info(f"Starting training: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if wandb_group:
        config["wandb_group"] = wandb_group

    train_args = build_cli_args(config)
    cmd = ["uv", "run", "src/lasernet/train.py"] + train_args

    exp_logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        exp_logger.error(f"Failed: {' '.join(cmd)}")
        raise RuntimeError(f"Training failed for {config_path}")

    exp_logger.info(f"Training completed: {config_path}")


def eval_and_predict_experiment(config_path: Path):
    """Run evaluate and predict steps sequentially for one experiment."""
    exp_logger = setup_experiment_logger(config_path)
    exp_logger.info(f"Starting eval & predict: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_args = build_cli_args({
        "network": config["network"],
        "field_type": config["field_type"],
        "loss": config.get("loss", "mse"),
        "num_workers": config.get("num_workers", 0),
        "batch_size": config.get("batch_size", 16),
        "seq_len": config.get("seq_len", 3),
    })
    predict_args = build_cli_args({
        "network": config["network"],
        "field_type": config["field_type"],
        "loss": config.get("loss", "mse"),
        "timestep": 18,
        "seq_len": config.get("seq_len", 3),
    })

    commands = [
        ["uv", "run", "src/lasernet/evaluate.py"] + eval_args,
        ["uv", "run", "src/lasernet/predict.py"] + predict_args,
    ]

    for cmd in commands:
        exp_logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            exp_logger.error(f"Failed: {' '.join(cmd)}")
            raise RuntimeError(f"Eval/Predict failed for {config_path}")

    exp_logger.info(f"Eval & Predict completed: {config_path}")


def main():
    num_workers = 1  # Run sequentially on single GPU

    # Generate W&B group name based on date
    wandb_group = datetime.now().strftime("seqlen-exp-%Y-%m-%d")
    logger.info(f"W&B group: {wandb_group}")

    # Phase 1: Train all experiments
    logger.info(f"\n{'='*60}")
    logger.info("Phase 1: Training sequence length experiments...")
    logger.info(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(train_experiment, Path(exp_config), wandb_group): exp_config
            for exp_config in EXPERIMENTS
        }

        for future in as_completed(futures):
            exp_config = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Training failed: {exp_config} - {e}")
                raise

    logger.info(f"\n{'='*60}")
    logger.info("Phase 2: Evaluating and predicting all experiments...")
    logger.info(f"{'='*60}\n")

    # Phase 2: Evaluate and predict sequentially
    for exp_config in EXPERIMENTS:
        try:
            eval_and_predict_experiment(Path(exp_config))
        except Exception as e:
            logger.error(f"Eval/Predict failed: {exp_config} - {e}")

    logger.info(f"\n{'='*60}")
    logger.info("All sequence length experiments completed!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

import subprocess
from pathlib import Path
import yaml
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

logger = logging.getLogger(__name__)

EXPERIMENTS = [
    "configs/experiments/cnn_temp_baseline.yaml",
    "configs/experiments/cnn_temp_combined.yaml",
    "configs/experiments/cnn_micro_baseline.yaml",
    "configs/experiments/cnn_micro_combined.yaml",
    "configs/experiments/transformer_temp_combined.yaml",
    "configs/experiments/transformer_micro_combined.yaml",
    "configs/experiments/cnn_temp_short_seq.yaml",
    "configs/experiments/cnn_micro_short_seq.yaml",
]

def build_cli_args(config: dict, extra_args: dict = None) -> list:
    """Convert config dict to CLI arguments.
    
    Args:
        config: Dictionary of config parameters
        extra_args: Additional arguments to override/add
        
    Returns:
        List of command-line arguments
    """
    args = []
    
    # Merge extra args if provided
    if extra_args:
        config = {**config, **extra_args}
    
    for key, value in config.items():
        # Convert underscores to hyphens for CLI
        cli_key = key.replace('_', '-')
        
        # Boolean flags
        if isinstance(value, bool):
            if value:
                args.append(f"--{cli_key}")
        # Other values
        elif value is not None:
            args.extend([f"--{cli_key}", str(value)])
    
    return args

def setup_experiment_logger(config_path: Path) -> logging.Logger:
    """Create a logger with file handler for this experiment."""
    exp_name = config_path.stem  # e.g., "cnn_temp_baseline"
    log_dir = Path("logs/experiments")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific logger
    exp_logger = logging.getLogger(exp_name)
    exp_logger.setLevel(logging.INFO)
    
    # File handler for this experiment
    log_file = log_dir / f"{exp_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler if not already present
    if not exp_logger.handlers:
        exp_logger.addHandler(file_handler)
    
    return exp_logger

def run_experiment(config_path: Path):
    """Run train -> evaluate -> predict pipeline for one experiment."""
    logger.info(f"Starting experiment: {config_path}")
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config: {config}")
    
    # Build commands with parsed arguments
    train_args = build_cli_args(config)
    eval_args = build_cli_args({
        "network": config["network"],
        "field_type": config["field_type"],
        "loss": config.get("loss", "mse"),
        "num_workers": config.get("num_workers", 0),
        "batch_size": config.get("batch_size", 16),
    })
    predict_args = build_cli_args({
        "network": config["network"],
        "field_type": config["field_type"],
        "loss": config.get("loss", "mse"),
        "timestep": 18,
    })
    
    commands = [
        ["uv", "run", "src/lasernet/train.py"] + train_args,
        ["uv", "run", "src/lasernet/evaluate.py"] + eval_args,
        ["uv", "run", "src/lasernet/predict.py"] + predict_args,
    ]
    
    for cmd in commands:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"Failed: {' '.join(cmd)}")
            raise RuntimeError(f"Experiment {config_path} failed")
    
    logger.info(f"Completed experiment: {config_path}")

def train_experiment(config_path: Path):
    """Run only the training step for one experiment."""
    exp_logger = setup_experiment_logger(config_path)
    exp_logger.info(f"Starting training: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_args = build_cli_args(config)
    cmd = ["uv", "run", "src/lasernet/train.py"] + train_args
    
    exp_logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        exp_logger.error(f"Failed: {' '.join(cmd)}")
        raise RuntimeError(f"Training failed for {config_path}")
    
    exp_logger.info(f"✓ Training completed: {config_path}")

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
    })
    predict_args = build_cli_args({
        "network": config["network"],
        "field_type": config["field_type"],
        "loss": config.get("loss", "mse"),
        "timestep": 18,
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
    
    exp_logger.info(f"✓ Eval & Predict completed: {config_path}")

def main():
    num_workers = 2  # Parallelize training across N GPUs
    
    # Phase 1: Train all experiments in parallel
    logger.info(f"\n{'='*60}")
    logger.info("Phase 1: Training all experiments in parallel...")
    logger.info(f"{'='*60}\n")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(train_experiment, Path(exp_config)): exp_config
            for exp_config in EXPERIMENTS
        }
        
        for future in as_completed(futures):
            exp_config = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"✗ Training failed: {exp_config} - {e}")
                raise
    
    logger.info(f"\n{'='*60}")
    logger.info("Phase 2: Evaluating and predicting all experiments...")
    logger.info(f"{'='*60}\n")
    
    # Phase 2: Evaluate and predict sequentially (after all training done)
    for exp_config in EXPERIMENTS:
        try:
            eval_and_predict_experiment(Path(exp_config))
        except Exception as e:
            logger.error(f"✗ Eval/Predict failed: {exp_config} - {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("All experiments completed!")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
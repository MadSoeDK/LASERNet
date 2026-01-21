````markdown
# lasernet

Spatiotemporal deep learning for predicting microstructure evolution in laser-based additive manufacturing

## Run
If on DTU HPC create symlinks to both data and models folder on the blackhole scratch drive. These folders use roughly 50 GB storage. 
```
ln -s "$BLACKHOLE/models" /zhome/b0/7/168550/Github/LASERNet/models
ln -s "$BLACKHOLE/data" /zhome/b0/7/168550/Github/LASERNet/data
```

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```

````

## Tensorboard on the HPC
Start tensorboard
```
make tensorboard
```

From you local machine, create SSH tunnel
```
ssh -L 6006:localhost:6006 your_username@hpc_login_node
```
Then open `http://localhost:6006` in your browser

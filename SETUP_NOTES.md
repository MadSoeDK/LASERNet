# Setup Notes

## Environment Variable: BLACKHOLE

The `BLACKHOLE` environment variable points to the data directory and is **automatically set** in the dataset code. You don't need to manually export it.

### Where it's set:

Both dataset files set it at the top of the file:

**[lasernet/dataset/loading.py](lasernet/dataset/loading.py)**:
```python
import os
os.environ["BLACKHOLE"] = "/dtu/blackhole/06/168550"
```

**[lasernet/dataset/preprocess_data.py](lasernet/dataset/preprocess_data.py)**:
```python
import os
os.environ["BLACKHOLE"] = "/dtu/blackhole/06/168550"
```

### Why this approach?

- **No manual setup needed**: Users don't have to remember to export the variable
- **Consistent across environments**: Works on HPC cluster, local machines, etc.
- **Self-contained**: Dataset code handles its own configuration

### If you need to change the data path:

Simply edit the value in both files:
```python
os.environ["BLACKHOLE"] = "/path/to/your/data"
```

## Running Tests

Before submitting to HPC, run the quick test:

```bash
# Test the microstructure prediction pipeline
python test_microstructure.py

# Or with uv
uv run test_microstructure.py
```

This will verify:
- Dataset loading works
- Model creation works
- Forward/backward passes work
- Training loop works

## HPC Submission

Once tests pass, submit the job:

```bash
bsub < batch/scripts/train_microstructure.sh
```

Check job status:
```bash
bjobs
```

View logs:
```bash
tail -f logs/lasernet_micro_<JOBID>.out
```

## Directory Structure

```
LASERNet/
├── lasernet/
│   ├── dataset/
│   │   ├── loading.py          # Sets BLACKHOLE env var
│   │   └── preprocess_data.py  # Sets BLACKHOLE env var
│   ├── model/
│   │   ├── CNN_LSTM.py         # Temperature prediction
│   │   └── MicrostructureCNN_LSTM.py  # Microstructure prediction
│   └── utils/
├── train.py                    # Temperature training
├── train_microstructure.py     # Microstructure training
├── test_microstructure.py      # Quick test script
└── batch/scripts/
    ├── train.sh                # Temperature HPC job
    └── train_microstructure.sh # Microstructure HPC job
```

## Data Location

The data is expected at:
```
$BLACKHOLE/Data/Alldata_withpoints_*.csv
```

Which resolves to:
```
/dtu/blackhole/06/168550/Data/Alldata_withpoints_*.csv
```

Files should be named like:
- `Alldata_withpoints_0.csv`
- `Alldata_withpoints_1.csv`
- `Alldata_withpoints_2.csv`
- etc.

Each file represents one timestep in the simulation.

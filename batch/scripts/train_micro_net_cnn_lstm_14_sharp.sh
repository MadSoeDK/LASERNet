#BSUB -o logs/micro-net_sharp_%J.out
#BSUB -e logs/micro-net_sharp_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the microstructure training script with SharpnessEnhancedLoss
python train_micro_net_cnn_lstm.py \
  --epochs 200 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 3 \
  --plane xz \
  --split-ratio "12,6,6" \
  --use-fast-loading \
  --use-weighted-loss \
  --loss-type sharpness \
  --gradient-weight 0.1 \
  --perceptual-weight 0.0 \
  --enable-solidification-sharp \
  --T-solidus 1560.0 \
  --T-liquidus 1620.0 \
  --weight-scale 0.1 \
  --base-weight 0.1


# How to Reduce Blurriness in Microstructure Predictions

## The Problem

Your predictions are blurry because **MSE (Mean Squared Error) loss** encourages the model to output the **average** of possible predictions. This is a fundamental limitation of pixel-wise losses:

- MSE penalizes any deviation from the target equally
- The safest prediction is the mean/average, which is blurry
- Sharp edges and fine details get smoothed out

## Solutions Implemented

I've added three new loss components to help reduce blurriness:

### 1. **Gradient Penalty Loss** (Most Important)
- **What it does**: Encourages sharp edges by maximizing spatial gradients
- **How it works**: Computes the difference between neighboring pixels and rewards larger gradients (sharper transitions)
- **Best for**: Recovering crisp grain boundaries and sharp features

### 2. **Perceptual Loss**
- **What it does**: Matches features instead of pixels
- **How it works**: Uses a CNN to extract multi-scale features and compares them using L1 loss
- **Best for**: Maintaining overall structure while avoiding pixel-exact averaging

### 3. **Sharpness Enhanced Loss**
- **What it does**: Combines all the above into one loss function
- **Components**: MSE (accuracy) + Gradient Penalty (sharpness) + Perceptual Loss (optional)

## How to Use

### Quick Start: Train with Sharpness Loss

```bash
python train_micro_net_cnn_lstm.py \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-3 \
    --seq-length 4 \
    --use-fast-loading \
    --use-weighted-loss \
    --loss-type sharpness \
    --gradient-weight 0.1 \
    --T-solidus 1000.0 \
    --T-liquidus 2500.0
```

### Recommended Settings for Different Scenarios

#### **Scenario 1: Maximum Sharpness** (Start Here!)
```bash
python train_micro_net_cnn_lstm.py \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-3 \
    --seq-length 4 \
    --use-fast-loading \
    --use-weighted-loss \
    --loss-type sharpness \
    --gradient-weight 0.2 \
    --perceptual-weight 0.0 \
    --T-solidus 1000.0 \
    --T-liquidus 2500.0
```

**Why this works:**
- `--gradient-weight 0.2`: Strong encouragement for sharp edges (higher = sharper)
- Combines solidification weighting with gradient penalty
- No perceptual loss to keep it simple

#### **Scenario 2: Balanced Sharpness + Accuracy**
```bash
python train_micro_net_cnn_lstm.py \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-3 \
    --seq-length 4 \
    --use-fast-loading \
    --use-weighted-loss \
    --loss-type sharpness \
    --gradient-weight 0.1 \
    --perceptual-weight 0.05 \
    --T-solidus 1000.0 \
    --T-liquidus 2500.0
```

**Why this works:**
- `--gradient-weight 0.1`: Moderate sharpness
- `--perceptual-weight 0.05`: Small perceptual component for feature matching
- Good balance between accuracy and sharpness

#### **Scenario 3: Very Aggressive Anti-Blur**
```bash
python train_micro_net_cnn_lstm.py \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-3 \
    --seq-length 4 \
    --use-fast-loading \
    --use-weighted-loss \
    --loss-type sharpness \
    --gradient-weight 0.5 \
    --perceptual-weight 0.1 \
    --T-solidus 1000.0 \
    --T-liquidus 2500.0
```

**Why this works:**
- `--gradient-weight 0.5`: Very strong sharpness enforcement
- `--perceptual-weight 0.1`: Additional perceptual matching
- **Warning**: May sacrifice some accuracy for sharpness

## Hyperparameter Guide

### `--gradient-weight` (Most Important!)
- **Default**: 0.1
- **Range**: 0.0 - 1.0
- **Effect**: Higher = sharper predictions, but may reduce overall accuracy
- **Recommended values**:
  - 0.05-0.1: Conservative (slight sharpness improvement)
  - 0.1-0.2: Moderate (good balance)
  - 0.2-0.5: Aggressive (maximum sharpness)

### `--perceptual-weight`
- **Default**: 0.0 (disabled)
- **Range**: 0.0 - 0.2
- **Effect**: Encourages feature-level similarity instead of pixel-level
- **Recommended values**:
  - 0.0: Disabled (simplest, good starting point)
  - 0.05-0.1: Moderate (helps with texture)
  - 0.1-0.2: Strong (may help with very blurry predictions)

### `--T-solidus` and `--T-liquidus`
- **Current values**: 1000.0 and 2500.0 (from your config)
- These define the solidification temperature range where weights are highest
- Adjust based on your material's actual phase transition temperatures

## Additional Strategies

### 1. **Reduce Learning Rate Over Time**
Blurriness can increase if the model oscillates. Use learning rate scheduling:

```python
# Add to train_micro_net_cnn_lstm.py
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# In validation loop:
scheduler.step(avg_val_loss)
```

### 2. **Increase Model Capacity**
A larger model can learn finer details. In [MicrostructureCNN_LSTM.py](lasernet/model/MicrostructureCNN_LSTM.py:34):

```python
# Current: hidden_channels = [16, 32, 64]
# Try: hidden_channels = [32, 64, 128]
```

### 3. **Post-Processing Sharpening** (Quick Fix)
If you need sharp predictions immediately, apply sharpening to existing predictions:

```python
from scipy.ndimage import laplace
import numpy as np

def sharpen_prediction(pred_rgb, amount=0.5):
    """Apply unsharp masking to sharpen prediction."""
    # Compute Laplacian (edge detection)
    laplacian = np.stack([laplace(pred_rgb[:,:,i]) for i in range(3)], axis=2)

    # Add back to original (sharpen)
    sharpened = pred_rgb - amount * laplacian

    # Clip to valid range
    return np.clip(sharpened, 0, 1)
```

### 4. **Use L1 Loss Instead of MSE**
L1 loss is less sensitive to outliers and may produce sharper results:

```python
# In train_micro_net_cnn_lstm.py, replace:
criterion = nn.MSELoss()
# With:
criterion = nn.L1Loss()
```

## Debugging Tips

### Check if Sharpness Loss is Working

Add to your training loop to monitor gradients:

```python
# After prediction
pred_grad_x = torch.abs(pred_micro[:, :, :, :-1] - pred_micro[:, :, :, 1:])
pred_grad_y = torch.abs(pred_micro[:, :, :-1, :] - pred_micro[:, :, 1:, :])
avg_gradient = (pred_grad_x.mean() + pred_grad_y.mean()) / 2
print(f"Average prediction gradient: {avg_gradient:.4f}")
```

**What to look for:**
- Gradient should **increase** over training epochs
- Higher gradient = sharper predictions
- Target gradient is typically 0.1-0.5 for IPF data

### Visualize Gradient Maps

```python
import matplotlib.pyplot as plt

def visualize_sharpness(pred, target, save_path):
    """Compare sharpness of prediction vs target."""
    pred_grad = np.abs(np.gradient(pred)[0])
    target_grad = np.abs(np.gradient(target)[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(target)
    axes[0].set_title('Ground Truth')
    axes[1].imshow(pred)
    axes[1].set_title(f'Prediction (grad: {pred_grad.mean():.3f})')
    axes[2].imshow(target_grad - pred_grad, cmap='RdBu_r')
    axes[2].set_title('Gradient Difference')
    plt.savefig(save_path)
```

## Expected Results

After retraining with sharpness loss:

1. **Grain boundaries** should be crisper and better defined
2. **Texture details** should be more visible
3. **MSE may slightly increase** (this is okay! Sharp != pixel-exact)
4. **Visual quality** should significantly improve

## Troubleshooting

### Problem: Predictions are too noisy/grainy
- **Solution**: Reduce `--gradient-weight` (try 0.05)
- May have set gradient weight too high

### Problem: Still blurry after training
- **Solution**: Increase `--gradient-weight` (try 0.3-0.5)
- Consider longer training (more epochs)
- Check if your model is underfitting

### Problem: Training is unstable
- **Solution**: Reduce learning rate `--lr 5e-4`
- Reduce gradient weight slightly
- Use gradient clipping

### Problem: Validation loss increases while training loss decreases
- **Solution**: You're overfitting. Add dropout or reduce model capacity
- Ensure you have enough training data

## Next Steps

1. **Start with Scenario 1** (Maximum Sharpness) and evaluate
2. If too sharp/noisy, reduce `--gradient-weight`
3. If still blurry, increase `--gradient-weight` or try Scenario 3
4. Compare predictions visually - MSE alone is not enough!

## Files Modified

- [lasernet/model/losses.py](lasernet/model/losses.py) - Added `SharpnessEnhancedLoss`, `GradientPenaltyLoss`, `PerceptualLoss`
- [train_micro_net_cnn_lstm.py](train_micro_net_cnn_lstm.py) - Added `--loss-type sharpness` option with gradient/perceptual weights

## References

- Gradient penalty is inspired by unsharp masking and edge-preserving losses
- Perceptual loss is based on feature matching (Johnson et al., 2016)
- The combination addresses the fundamental limitation of MSE loss in generative tasks

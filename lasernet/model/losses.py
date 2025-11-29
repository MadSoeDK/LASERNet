"""
Custom loss functions for microstructure prediction.

Implements solidification front proximity weighting to emphasize
the challenging phase transition region where metal goes from
molten to solid state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SolidificationWeightedMSELoss(nn.Module):
    """
    MSE loss with spatial weighting based on proximity to solidification front.

    The solidification front (molten→solid transition) is the most challenging
    region to predict because:
    1. Microstructure is actively forming (not static like solid or zero like molten)
    2. Small temperature changes cause large microstructure changes
    3. Physical theory requires capturing Moore neighborhood dependencies

    This loss applies higher weights to pixels near the solidification temperature
    range, forcing the model to focus on getting the phase transition right.

    Parameters:
        T_solidus: Solidus temperature (fully solid below this) in Kelvin
        T_liquidus: Liquidus temperature (fully liquid above this) in Kelvin
        weight_type: Type of weighting function ('gaussian', 'linear', 'exponential')
        weight_scale: Scale factor for weight curve (higher = more focused on front)
        base_weight: Minimum weight for regions outside solidification zone (0-1)
    """

    def __init__(
        self,
        T_solidus: float = 1400.0,
        T_liquidus: float = 1500.0,
        weight_type: str = "gaussian",
        weight_scale: float = 0.1,
        base_weight: float = 0.1,
    ):
        super().__init__()

        assert T_solidus < T_liquidus, "Solidus must be less than liquidus"
        assert weight_type in ["gaussian", "linear", "exponential"], \
            f"Invalid weight_type: {weight_type}"
        assert 0.0 <= base_weight <= 1.0, "base_weight must be in [0, 1]"

        self.T_solidus = T_solidus
        self.T_liquidus = T_liquidus
        self.weight_type = weight_type
        self.weight_scale = weight_scale
        self.base_weight = base_weight

        # Temperature normalization range (from model's normalization)
        self.temp_min = 300.0
        self.temp_max = 2000.0

    def _compute_weights(
        self,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spatial weight map based on temperature.

        Args:
            temperature: Temperature tensor [B, H, W] (normalized or unnormalized)
            mask: Valid region mask [B, H, W]

        Returns:
            weight: Spatial weight map [B, H, W]
        """
        # Check if temperature is normalized (values in [0, 1])
        if temperature.min() >= 0.0 and temperature.max() <= 1.0:
            # Denormalize temperature
            temp = temperature * (self.temp_max - self.temp_min) + self.temp_min
        else:
            temp = temperature

        # Normalize temperature to solidification range [0, 1]
        # 0 = solidus (fully solid), 1 = liquidus (fully liquid)
        temp_normalized = (temp - self.T_solidus) / (self.T_liquidus - self.T_solidus)
        temp_normalized = torch.clamp(temp_normalized, 0.0, 1.0)

        # Compute weights based on distance from mid-point (0.5)
        # Mid-point = where solidification is most active
        if self.weight_type == "gaussian":
            # Gaussian centered at 0.5 (peak weight at mid-solidification)
            distance = (temp_normalized - 0.5) ** 2
            weight = torch.exp(-distance / self.weight_scale)

        elif self.weight_type == "linear":
            # Linear decay from center
            distance = torch.abs(temp_normalized - 0.5)
            weight = 1.0 - torch.clamp(distance / 0.5, 0.0, 1.0)

        elif self.weight_type == "exponential":
            # Exponential decay from center
            distance = torch.abs(temp_normalized - 0.5)
            weight = torch.exp(-distance / self.weight_scale)

        else:
            raise ValueError(f"Invalid weight_type: {self.weight_type}")

        # Scale to [base_weight, 1.0] range
        weight = self.base_weight + (1.0 - self.base_weight) * weight

        # Apply valid region mask
        weight = weight * mask

        return weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predicted microstructure [B, C, H, W] (C=9 IPF channels)
            target: Target microstructure [B, C, H, W]
            temperature: Temperature field [B, H, W] or [B, 1, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar weighted MSE loss
        """
        # Handle temperature with channel dimension
        if temperature.dim() == 4:  # [B, 1, H, W]
            temperature = temperature.squeeze(1)  # [B, H, W]

        # Compute spatial weights
        weight = self._compute_weights(temperature, mask)  # [B, H, W]

        # Compute element-wise MSE
        mse = (pred - target) ** 2  # [B, C, H, W]

        # Expand weights for all channels
        weight_expanded = weight.unsqueeze(1)  # [B, 1, H, W]

        # Apply weights
        weighted_mse = mse * weight_expanded  # [B, C, H, W]

        # Compute mean loss (normalized by total weight)
        total_weight = weight.sum() * pred.size(1)  # sum of weights × num channels

        if total_weight > 0:
            loss = weighted_mse.sum() / total_weight
        else:
            # Fallback to unweighted MSE if no valid pixels
            loss = mse.mean()

        return loss

    def get_weight_map(
        self,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the weight map for visualization/debugging.

        Args:
            temperature: Temperature field [B, H, W] or [B, 1, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            weight: Weight map [B, H, W]
        """
        if temperature.dim() == 4:
            temperature = temperature.squeeze(1)

        return self._compute_weights(temperature, mask)


class CombinedLoss(nn.Module):
    """
    Combines multiple loss terms with configurable weights.

    Useful for balancing solidification front weighting with
    global accuracy.

    Example:
        # 70% weight on solidification front, 30% on global MSE
        loss_fn = CombinedLoss(
            solidification_weight=0.7,
            global_weight=0.3,
        )
    """

    def __init__(
        self,
        solidification_weight: float = 0.7,
        global_weight: float = 0.3,
        T_solidus: float = 1400.0,
        T_liquidus: float = 1500.0,
        weight_type: str = "gaussian",
        weight_scale: float = 0.1,
        base_weight: float = 0.1,
        return_components: bool = False,
    ):
        super().__init__()

        self.solidification_weight = solidification_weight
        self.global_weight = global_weight
        self.return_components = return_components

        self.solidification_loss = SolidificationWeightedMSELoss(
            T_solidus=T_solidus,
            T_liquidus=T_liquidus,
            weight_type=weight_type,
            weight_scale=weight_scale,
            base_weight=base_weight,
        )

        self.global_loss = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Compute combined loss.

        Args:
            pred: Predicted microstructure [B, C, H, W]
            target: Target microstructure [B, C, H, W]
            temperature: Temperature field [B, H, W] or [B, 1, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            If return_components is False:
                loss: Scalar combined loss
            If return_components is True:
                tuple: (total_loss, solidification_loss, global_loss)
        """
        # Solidification front weighted loss
        solid_loss = self.solidification_loss(pred, target, temperature, mask)

        # Global MSE loss (only on valid pixels)
        mask_expanded = mask.unsqueeze(1).expand_as(target)
        global_loss = self.global_loss(
            pred[mask_expanded],
            target[mask_expanded],
        )

        # Combine
        total_loss = (
            self.solidification_weight * solid_loss +
            self.global_weight * global_loss
        )

        if self.return_components:
            return total_loss, solid_loss, global_loss
        else:
            return total_loss


class GradientPenaltyLoss(nn.Module):
    """
    Gradient penalty loss to encourage sharp edges.

    This loss penalizes smoothness by computing the L1 norm of spatial gradients.
    Minimizing this loss encourages sharper boundaries and more defined features.

    However, it should be used with negative weight or as a regularization term
    since we want to MAXIMIZE gradients (sharp edges), not minimize them.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty (sharpness) loss.

        Args:
            pred: Predicted microstructure [B, C, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar gradient penalty
        """
        # Compute spatial gradients
        grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])  # [B, C, H, W-1]
        grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])  # [B, C, H-1, W]

        # Apply mask (erosion to handle boundaries)
        mask_x = mask[:, :, :-1] * mask[:, :, 1:]  # [B, H, W-1]
        mask_y = mask[:, :-1, :] * mask[:, 1:, :]  # [B, H-1, W]

        # Expand masks for all channels
        mask_x = mask_x.unsqueeze(1)  # [B, 1, H, W-1]
        mask_y = mask_y.unsqueeze(1)  # [B, 1, H-1, W]

        # Compute mean gradient magnitude (only in valid regions)
        grad_x_masked = grad_x * mask_x
        grad_y_masked = grad_y * mask_y

        total_x = grad_x_masked.sum()
        total_y = grad_y_masked.sum()
        count_x = mask_x.sum() * pred.size(1)
        count_y = mask_y.sum() * pred.size(1)

        # Average gradient magnitude
        if count_x > 0 and count_y > 0:
            avg_grad = (total_x / count_x + total_y / count_y) / 2
        else:
            avg_grad = torch.tensor(0.0, device=pred.device)

        # Return negative gradient (we want to maximize, so minimize negative)
        # Or return positive if using as a penalty term
        return -self.weight * avg_grad


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using feature similarity instead of pixel-wise similarity.

    Uses a simple CNN to extract features at multiple scales and compares
    them using L1 loss. This encourages perceptually similar outputs
    rather than pixel-exact matches, which helps avoid blurriness.

    Simpler than VGG-based perceptual loss but suitable for scientific data.
    """

    def __init__(self, feature_channels: list = [16, 32, 64]):
        super().__init__()

        # Build feature extractor
        layers = []
        in_channels = 3  # RGB (IPF-X) input

        for out_channels in feature_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # Freeze parameters (no training)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted microstructure [B, 9, H, W]
            target: Target microstructure [B, 9, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar perceptual loss
        """
        # Extract RGB channels (IPF-X: channels 0-2)
        pred_rgb = pred[:, :3]    # [B, 3, H, W]
        target_rgb = target[:, :3]  # [B, 3, H, W]

        # Extract features
        pred_features = self.feature_extractor(pred_rgb)
        target_features = self.feature_extractor(target_rgb)

        # Compute L1 loss on features
        loss = nn.functional.l1_loss(pred_features, target_features)

        return loss


class SharpnessEnhancedLoss(nn.Module):
    """
    Combined loss function designed to reduce blurriness.

    Combines:
    1. MSE loss (or weighted MSE) for overall accuracy
    2. Gradient penalty (negative) to encourage sharp edges
    3. Perceptual loss (optional) for feature-level similarity

    This addresses the fundamental issue that MSE alone produces blurry outputs
    by averaging predictions.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        gradient_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        use_solidification_weighting: bool = False,
        use_l1_loss: bool = False,
        use_charbonnier: bool = False,
        T_solidus: float = 1000.0,
        T_liquidus: float = 2500.0,
        weight_scale: float = 0.1,
        base_weight: float = 0.1,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        self.perceptual_weight = perceptual_weight
        self.use_l1_loss = use_l1_loss
        self.use_charbonnier = use_charbonnier

        # Base reconstruction loss
        if use_solidification_weighting:
            self.mse_loss = SolidificationWeightedMSELoss(
                T_solidus=T_solidus,
                T_liquidus=T_liquidus,
                weight_type="gaussian",
                weight_scale=weight_scale,
                base_weight=base_weight,
            )
            self.needs_temperature = True
        elif use_charbonnier:
            self.mse_loss = CharbonnierLoss()
            self.needs_temperature = False
        elif use_l1_loss:
            self.mse_loss = L1Loss()
            self.needs_temperature = False
        else:
            self.mse_loss = nn.MSELoss()
            self.needs_temperature = False

        # Gradient penalty (for sharpness)
        self.gradient_loss = GradientPenaltyLoss(weight=1.0)

        # Perceptual loss (optional)
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        temperature: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute sharpness-enhanced loss.

        Args:
            pred: Predicted microstructure [B, 9, H, W]
            target: Target microstructure [B, 9, H, W]
            temperature: Temperature field [B, H, W] or [B, 1, H, W] (required if using solidification weighting)
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar combined loss
        """
        total_loss = 0.0

        # 1. MSE loss (accuracy)
        if self.needs_temperature:
            if temperature is None:
                raise ValueError("Temperature required for solidification-weighted loss")
            mse = self.mse_loss(pred, target, temperature, mask)
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).expand_as(target)
                mse = nn.functional.mse_loss(pred[mask_expanded], target[mask_expanded])
            else:
                mse = self.mse_loss(pred, target)

        total_loss += self.mse_weight * mse

        # 2. Gradient penalty (sharpness)
        if self.gradient_weight > 0:
            grad_penalty = self.gradient_loss(pred, mask if mask is not None else torch.ones_like(pred[:, 0]))
            total_loss += self.gradient_weight * grad_penalty

        # 3. Perceptual loss (optional)
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target, mask if mask is not None else torch.ones_like(pred[:, 0]))
            total_loss += self.perceptual_weight * perceptual

        return total_loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation (TV) loss - another approach to encourage smoothness while preserving edges.

    Unlike gradient penalty which maximizes gradients, TV loss encourages
    piecewise constant regions (smooth within grains, sharp at boundaries).
    This is commonly used in image denoising and super-resolution.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            pred: Predicted microstructure [B, C, H, W]

        Returns:
            loss: Scalar TV loss
        """
        # Compute differences along height and width
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        # Sum of absolute differences
        tv_loss = diff_h.sum() + diff_w.sum()

        return self.weight * tv_loss


class L1Loss(nn.Module):
    """
    L1 loss (Mean Absolute Error) as an alternative to MSE.

    L1 loss is more robust to outliers and tends to produce sharper
    results than L2/MSE loss because it doesn't square errors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute L1 loss.

        Args:
            pred: Predicted microstructure [B, C, H, W]
            target: Target microstructure [B, C, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar L1 loss
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(target)
            return nn.functional.l1_loss(pred[mask_expanded], target[mask_expanded])
        else:
            return nn.functional.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1) - a differentiable variant of L1 loss.

    This is used in many image restoration tasks and produces sharper
    results than MSE while being smoother than pure L1.

    Formula: sqrt(||pred - target||^2 + epsilon^2)
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute Charbonnier loss.

        Args:
            pred: Predicted microstructure [B, C, H, W]
            target: Target microstructure [B, C, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar Charbonnier loss
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(target)
            return loss[mask_expanded].mean()
        else:
            return loss.mean()
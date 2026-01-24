import torch
import torch.nn as nn
import torch.nn.functional as F

from lasernet.laser_types import T_SOLIDUS, T_LIQUIDUS


class GradientWeightedMSELoss(nn.Module):
    """
    MSE loss with spatial weighting based on temperature gradients.

    Applies higher weights to regions with sharp temperature gradients,
    which correspond to the solidification front and melt pool boundaries.
    This forces the model to accurately capture these critical transition regions.

    Parameters:
        gradient_weight: How much to weight high-gradient regions (default 10.0)
        base_weight: Minimum weight for low-gradient regions (default 0.1)
        gradient_threshold: Gradients below this are considered "low" (default 0.01)
        normalize_gradients: Whether to normalize gradients to [0, 1] range
    """

    def __init__(
        self,
        gradient_weight: float = 10.0,
        base_weight: float = 0.1,
        gradient_threshold: float = 0.01,
        normalize_gradients: bool = True,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.base_weight = base_weight
        self.gradient_threshold = gradient_threshold
        self.normalize_gradients = normalize_gradients

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _compute_gradient_magnitude(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradient magnitude using Sobel operators.

        Args:
            field: Input field [B, H, W] or [B, 1, H, W]

        Returns:
            gradient_mag: Gradient magnitude [B, H, W]
        """
        if field.dim() == 3:
            field = field.unsqueeze(1)  # [B, 1, H, W]

        # Ensure kernels are on same device and dtype
        sobel_x = self.sobel_x.to(field.device, field.dtype)
        sobel_y = self.sobel_y.to(field.device, field.dtype)

        # Compute gradients with padding to maintain size
        grad_x = F.conv2d(field, sobel_x, padding=1)
        grad_y = F.conv2d(field, sobel_y, padding=1)

        # Gradient magnitude
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).squeeze(1)  # [B, H, W]

        return gradient_mag

    def _compute_weights(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spatial weight map based on target gradients.

        Args:
            target: Target field [B, C, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            weight: Spatial weight map [B, H, W]
        """
        # Use first channel (temperature) for gradient computation
        # For temperature-only models, C=1
        # For combined models, first channel is temperature
        field = target[:, 0, :, :]  # [B, H, W]

        gradient_mag = self._compute_gradient_magnitude(field)  # [B, H, W]

        if self.normalize_gradients:
            # Normalize per-batch to [0, 1]
            grad_min = gradient_mag.amin(dim=(1, 2), keepdim=True)
            grad_max = gradient_mag.amax(dim=(1, 2), keepdim=True)
            grad_range = grad_max - grad_min + 1e-8
            gradient_mag = (gradient_mag - grad_min) / grad_range

        # Create weight map: high weight for high gradients
        # Linear interpolation between base_weight and gradient_weight
        weight = self.base_weight + (self.gradient_weight - self.base_weight) * gradient_mag

        # Apply valid region mask
        weight = weight * mask.float()

        return weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient-weighted MSE loss.

        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, C, H, W]
            temperature: Temperature field [B, H, W] (unused, kept for API compatibility)
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar weighted MSE loss
        """
        # Compute spatial weights based on target gradients
        weight = self._compute_weights(target, mask)  # [B, H, W]

        # Compute element-wise MSE
        mse = (pred - target) ** 2  # [B, C, H, W]

        # Expand weights for all channels
        weight_expanded = weight.unsqueeze(1)  # [B, 1, H, W]

        # Apply weights
        weighted_mse = mse * weight_expanded  # [B, C, H, W]

        # Compute mean loss (normalized by total weight)
        total_weight = weight.sum() * pred.size(1)

        if total_weight > 0:
            loss = weighted_mse.sum() / total_weight
        else:
            loss = mse.mean()

        return loss


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
        T_solidus: float = T_SOLIDUS,
        T_liquidus: float = T_LIQUIDUS,
        weight_type: str = "gaussian",
        weight_scale: float = 0.1,
        base_weight: float = 0.1,
        temp_min: float = 0.0,
        temp_max: float = 4644.0,
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

        # Temperature normalization range (should match DataNormalizer stats)
        self.temp_min = temp_min
        self.temp_max = temp_max

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

        # Create binary mask for solidification range
        in_solidification_range = (temp >= self.T_solidus) & (temp <= self.T_liquidus)

        # Weight = 1.0 everywhere, but only compute loss in solidification region
        # base_weight controls whether to include regions outside solidification
        weight = torch.where(
            in_solidification_range,
            torch.ones_like(temp),  # weight = 1.0 in solidification range
            torch.ones_like(temp) * self.base_weight  # base_weight outside (0.0 = exclude)
        )

        # Apply valid region mask: regions outside mask get base_weight reduction
        # but are NOT completely zeroed out. This prevents model from learning to 
        # output zeros in cold regions while still prioritizing the heated area.
        mask_float = mask.float()
        weight = torch.where(
            mask_float > 0,
            weight,  # Full weight in valid region
            weight * self.base_weight  # Reduced weight outside valid region
        )

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
        # 50% weight on solidification front, 50% on global MSE
        loss_fn = CombinedLoss(
            solidification_weight=0.5,
            global_weight=0.5,
        )
    """

    def __init__(
        self,
        solidification_weight: float = 0.5,
        global_weight: float = 0.5,
        T_solidus: float = T_SOLIDUS,
        T_liquidus: float = T_LIQUIDUS,
        weight_type: str = "gaussian",
        weight_scale: float = 0.1,
        base_weight: float = 0.1,
        temp_min: float = 0.0,
        temp_max: float = 4644.0,
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
            temp_min=temp_min,
            temp_max=temp_max,
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
        mask_expanded = mask.bool().unsqueeze(1).expand_as(target)
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
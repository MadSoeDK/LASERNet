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

        # Create binary mask for solidification range
        in_solidification_range = (temp_normalized >= 0.0) & (temp_normalized <= 1.0)

        # Uniform weight across full solidification range
        weight = torch.where(
            in_solidification_range,
            torch.ones_like(temp_normalized),  # weight = 1.0 in solidification range
            torch.ones_like(temp_normalized) * self.base_weight  # base_weight outside
        )

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

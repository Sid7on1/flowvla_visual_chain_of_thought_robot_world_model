import torch
import numpy as np
import cv2
from PIL import Image
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpticalFlowEncoder:
    """
    OpticalFlowEncoder converts 2-channel optical flow vectors to RGB images using polar coordinate mapping.
    It also provides utilities for normalizing and denormalizing magnitudes, and visualizing flow.

    ...

    Attributes
    ----------
    device : str or torch.device
        The device on which the tensors will be allocated.

    Methods
    -------
    flow_to_rgb(flow_uv, magnitude_threshold=0.0)
        Convert optical flow UV vectors to RGB images.
    polar_to_rgb(angle, magnitude, max_radius=None, cmap='hsv')
        Convert polar coordinates to RGB images using a colormap.
    normalize_magnitude(flow_uv, min_magnitude=1e-4)
        Normalize the magnitude of optical flow vectors.
    denormalize_magnitude(flow_uv, original_shape)
        Denormalize the magnitude of optical flow vectors.
    visualize_flow(flow_uv, original_shape)
        Visualize optical flow vectors as an RGB image.

    """

    def __init__(self, device='cpu'):
        self.device = device

    def flow_to_rgb(self, flow_uv: torch.Tensor, magnitude_threshold: float = 0.0) -> torch.Tensor:
        """
        Convert optical flow UV vectors to RGB images.

        Parameters
        ----------
        flow_uv : torch.Tensor
            Optical flow UV vectors of shape (batch_size, 2, height, width).
        magnitude_threshold : float, optional
            Threshold to apply to the flow magnitude. Default is 0.0.

        Returns
        -------
        rgb_image : torch.Tensor
            RGB image representation of the optical flow of shape (batch_size, 3, height, width).

        Raises
        ------
        ValueError
            If the input tensor has an incorrect number of channels or the magnitude threshold is negative.

        """
        if flow_uv.shape[1] != 2:
            raise ValueError("Input tensor must have 2 channels for UV components.")
        if magnitude_threshold < 0:
            raise ValueError("Magnitude threshold must be non-negative.")

        # Separate U and V components
        flow_u = flow_uv[..., 0]
        flow_v = flow_uv[..., 1]

        # Compute flow magnitude and angle
        flow_magnitude = torch.sqrt(flow_u ** 2 + flow_v ** 2)
        flow_angle = torch.atan2(flow_v, flow_u)

        # Apply magnitude threshold
        flow_magnitude = torch.where(flow_magnitude > magnitude_threshold, flow_magnitude, torch.zeros_like(flow_magnitude))

        # Convert polar coordinates to RGB using HSV color space
        rgb_image = self.polar_to_rgb(flow_angle, flow_magnitude)

        return rgb_image

    def polar_to_rgb(self, angle: torch.Tensor, magnitude: torch.Tensor, max_radius: int = None, cmap: str = 'hsv') -> torch.Tensor:
        """
        Convert polar coordinates to RGB images using a colormap.

        Parameters
        ----------
        angle : torch.Tensor
            Angle values in radians.
        magnitude : torch.Tensor
            Magnitude values.
        max_radius : int, optional
            Maximum radius for the polar coordinate system. If None, the maximum value of 'magnitude' is used. Default is None.
        cmap : str, optional
            Colormap to use. Default is 'hsv'.

        Returns
        -------
        rgb_image : torch.Tensor
            RGB image representation of the polar coordinates.

        Raises
        ------
        ValueError
            If the colormap is not supported.

        """
        if max_radius is None:
            max_radius = int(magnitude.max())

        # Create coordinate grids
        height, width = angle.shape[-2:]
        x_coord = torch.arange(width, device=self.device).repeat(height, 1)
        y_coord = torch.arange(height, device=self.device).repeat(width, 1).transpose(0, 1)

        # Convert angle and magnitude to Cartesian coordinates
        x_cart = (magnitude * torch.cos(angle)).unsqueeze(1)
        y_cart = (magnitude * torch.sin(angle)).unsqueeze(1)

        # Clip coordinates within the image bounds
        x_cart = torch.clamp(x_coord + x_cart, 0, width - 1)
        y_cart = torch.clamp(y_coord + y_cart, 0, height - 1)

        # Create RGB image
        rgb_image = torch.zeros((3, height, width), device=self.device)
        for c in range(3):
            channel = cv2.remap(y_cart, x_cart, None, cv2.INTER_NEAREST, borderValue=max_radius if c == 0 else 0)
            rgb_image[c] = torch.from_numpy(channel).to(self.device)

        # Convert to HSV color space and apply colormap
        rgb_image = cv2.cvtColor(rgb_image.cpu().numpy(), cv2.COLOR_BGR2RGB)
        if cmap == 'hsv':
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            hsv_image[:, :, 0] = hsv_image[:, :, 0] * 0.5 / max_radius
            hsv_image[:, :, 1] = 255
            hsv_image[:, :, 2] = 255
            rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        elif cmap == 'magma':
            rgb_image = plt.cm.magma(rgb_image)[:, :, :3]
        else:
            raise ValueError("Unsupported colormap. Choose from 'hsv' or 'magma'.")

        rgb_image = torch.from_numpy(rgb_image).to(self.device)

        return rgb_image

    def normalize_magnitude(self, flow_uv: torch.Tensor, min_magnitude: float = 1e-4) -> torch.Tensor:
        """
        Normalize the magnitude of optical flow vectors.

        Parameters
        ----------
        flow_uv : torch.Tensor
            Optical flow UV vectors of shape (batch_size, 2, height, width).
        min_magnitude : float, optional
            Minimum magnitude value. Default is 1e-4.

        Returns
        -------
        normalized_flow_uv : torch.Tensor
            Normalized optical flow UV vectors.

        """
        flow_magnitude = torch.sqrt(torch.sum(flow_uv ** 2, dim=1, keepdim=True))
        max_magnitude = flow_magnitude.max()

        normalized_flow_uv = flow_uv / (max_magnitude + 1e-8)
        normalized_flow_uv *= min_magnitude

        return normalized_flow_uv

    def denormalize_magnitude(self, flow_uv: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Denormalize the magnitude of optical flow vectors.

        Parameters
        ----------
        flow_uv : torch.Tensor
            Normalized optical flow UV vectors of shape (batch_size, 2, height, width).
        original_shape : tuple
            Original shape of the optical flow before normalization.

        Returns
        -------
        denormalized_flow_uv : torch.Tensor
            Denormalized optical flow UV vectors.

        """
        batch_size, _, height, width = original_shape
        original_flow_uv = torch.zeros(original_shape, device=self.device)

        # Reshape to 2D for broadcasting
        original_flow_uv = original_flow_uv.view(batch_size, 2, -1)
        flow_uv = flow_uv.view(batch_size, 2, -1)

        # Denormalize magnitude
        original_flow_uv[..., 0] = flow_uv[..., 0] * original_flow_uv[..., 0].sum(-1, keepdim=True) / flow_uv[..., 0].sum(-1, keepdim=True)
        original_flow_uv[..., 1] = flow_uv[..., 1] * original_flow_uv[..., 1].sum(-1, keepdim=True) / flow_uv[..., 1].sum(-1, keepdim=True)

        # Restore original shape
        original_flow_uv = original_flow_uv.view(batch_size, 2, height, width)

        return original_flow_uv

    def visualize_flow(self, flow_uv: torch.Tensor, original_shape: tuple) -> Image:
        """
        Visualize optical flow vectors as an RGB image.

        Parameters
        ----------
        flow_uv : torch.Tensor
            Optical flow UV vectors of shape (batch_size, 2, height, width).
        original_shape : tuple
            Original shape of the optical flow before processing.

        Returns
        -------
        rgb_image : Image
            RGB image representation of the optical flow.

        """
        flow_uv = self.denormalize_magnitude(flow_uv, original_shape)
        rgb_image = self.flow_to_rgb(flow_uv)
        rgb_image = rgb_image.cpu().numpy().astype(np.uint8)
        rgb_image = np.moveaxis(rgb_image, 1, -1)
        rgb_image = Image.fromarray(rgb_image[0])

        return rgb_image

# Example usage
if __name__ == "__main__":
    flow_uv = torch.randn(1, 2, 32, 32)
    encoder = OpticalFlowEncoder()
    rgb_image = encoder.flow_to_rgb(flow_uv)
    normalized_flow_uv = encoder.normalize_magnitude(flow_uv)
    denormalized_flow_uv = encoder.denormalize_magnitude(normalized_flow_uv, flow_uv.shape)
    visualized_flow = encoder.visualize_flow(flow_uv, flow_uv.shape)

    logger.info(f"Original flow UV shape: {flow_uv.shape}")
    logger.info(f"RGB image shape: {rgb_image.shape}")
    logger.info(f"Normalized flow UV shape: {normalized_flow_uv.shape}")
    logger.info(f"Denormalized flow UV shape: {denormalized_flow_uv.shape}")
    logger.info("Visualized flow saved as 'visualized_flow.png'")

    # Save visualized flow
    visualized_flow.save('visualized_flow.png')
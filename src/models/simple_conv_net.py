import torch
from torch import nn

from src.models._base_model import BaseModel
from src.utilities.utils import get_normalization_layer


class ConvBlock(nn.Module):
    """ A simple convolutional block with BatchNorm and GELU activation. """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 residual: bool = False,
                 net_normalization: str = 'batch_norm',
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = get_normalization_layer(net_normalization, out_channels, num_groups=32)
        self.norm = nn.BatchNorm2d(out_channels)  # a normalization layer for improved/more stable training
        self.activation = nn.GELU()  # a non-linearity
        self.residual = residual and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.residual:
            x = x + residual
        return x


# The model:
class SimpleConvNet(BaseModel):
    """ A simple convolutional network. """

    def __init__(self,
                 dim: int,
                 net_normalization: str = 'batch_norm',
                 residual=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dim = channels_hidden
        # Define the convolutional layers
        kwargs = dict(net_normalization=net_normalization)
        self.conv1 = ConvBlock(self.num_input_channels, dim, kernel_size=7, padding=3, **kwargs)
        self.conv2 = ConvBlock(dim, dim, kernel_size=3, padding=1, residual=residual, **kwargs)
        self.conv3 = ConvBlock(dim, dim, kernel_size=3, padding=1, residual=residual, **kwargs)
        self.head = nn.Conv2d(dim, self.num_output_channels, kernel_size=1, padding=0)

        # Example input
        self.example_input_array = torch.randn(1, self.num_input_channels, self.spatial_dims[0], self.spatial_dims[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a batch of images of shape (batch_size, channels_in, height, width)

        Returns:
            y: a batch of images of shape (batch_size, channels_out, height, width)
        """
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.head(h3)
        return h4

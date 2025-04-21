import torch
import torch.nn as nn
class sampleMLP(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.layernorm = nn.LayerNorm(2048)

        # Token-wise MLP
        self.mlp = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2048)
        )

        # Conv over sequence: Conv1D over dim=469 (seq_len)
        conv_kernel_size = 3
        self.conv = nn.Conv1d(
            in_channels=2048,
            out_channels=2048,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )

    def forward(self, x):
        x = self.layernorm(x)             # (B, 469, 2048)
        x_mlp = self.mlp(x)               # (B, 469, 2048)

        x_conv = x.transpose(1, 2)        # (B, 2048, 469)
        x_conv = self.conv(x_conv)        # (B, 2048, 469)
        x_conv = x_conv.transpose(1, 2)   # (B, 469, 2048)

        return x_mlp + x_conv             # (B, 469, 2048)


if __name__ == "__main__":
    model = sampleMLP()
    x = torch.randn(1, 469, 2048)
    out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "Output shape doesn't match input shape!"
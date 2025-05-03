import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block RB(N, W, D):
      - N: number of channels
      - W: kernel size
      - D: dilation
    Diagram:
      BatchNorm -> ReLU -> Conv(N, W, D)
      BatchNorm -> ReLU -> Conv(N, W, D)
      Skip connection adds input to output
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # First BN -> ReLU -> Conv
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        # Second BN -> ReLU -> Conv
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # Residual connection
        return x + out


class SpliceAI_10k(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=3):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1   = nn.Sequential(*[ResidualBlock(mid_channels, 11, 1) for _ in range(4)])
        self.conv11_4 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_4   = nn.Sequential(*[ResidualBlock(mid_channels, 11, 4) for _ in range(4)])
        self.conv21_10= nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb21_10  = nn.Sequential(*[ResidualBlock(mid_channels, 21, 10) for _ in range(4)])
        self.conv41_25= nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb41_25  = nn.Sequential(*[ResidualBlock(mid_channels, 41, 25) for _ in range(4)])
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)

        # Two convs per block, so each ResidualBlock adds 2*(kernel_size−1)*dilation.
        # Sum over all 4 blocks in each of the four groups:
        rf  = 1  # initial 1×1 conv has RF=1
        rf += 4 * 2 * (11 - 1) * 1    # first group
        rf += 4 * 2 * (11 - 1) * 4    # second group
        rf += 4 * 2 * (21 - 1) * 10   # third group
        rf += 4 * 2 * (41 - 1) * 25   # fourth group
        # total rf = 1 + 80 + 320 + 1600 + 8000 = 10001
        self.receptive_field = rf
        # we want to drop rf//2 at each end -only predict the middle 5000 bases
        self.crop = (rf - 1) // 2      # = 5000

    def forward(self, x):
        x = self.initial_conv(x)
        # block 1
        skip = self.conv11_1(x)
        x  = self.rb11_1(x)
        # block 2
        skip += self.conv11_4(x)
        x  = self.rb11_4(x)
        # block 3
        skip += self.conv21_10(x)
        x  = self.rb21_10(x)
        # block 4
        skip += self.conv41_25(x)
        x  = self.rb41_25(x)
        # final projections
        x = self.final_conv1(x) + skip
        x = self.final_conv2(x)
        # x = F.softmax(x, dim=1)  # since CrossEntropyLoss is used, we don't need to apply softmax here
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x

class SpliceAI_2k(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=3):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1   = nn.Sequential(*[ResidualBlock(mid_channels, 11, 1) for _ in range(4)])
        self.conv11_4 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_4   = nn.Sequential(*[ResidualBlock(mid_channels, 11, 4) for _ in range(4)])
        self.conv21_10= nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb21_10  = nn.Sequential(*[ResidualBlock(mid_channels, 21, 10) for _ in range(4)])
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)

        # Two convs per block, so each ResidualBlock adds 2*(kernel_size−1)*dilation.
        # Sum over all 4 blocks in each of the four groups:
        rf  = 1  # initial 1×1 conv has RF=1
        rf += 4 * 2 * (11 - 1) * 1    # first group
        rf += 4 * 2 * (11 - 1) * 4    # second group
        rf += 4 * 2 * (21 - 1) * 10   # third group
        # total rf = 1 + 80 + 320 + 1600 = 2001
        self.receptive_field = rf
        # we want to drop rf//2 at each end -only predict the middle 5000 bases
        self.crop = (rf - 1) // 2      # = 1000

    def forward(self, x):
        x = self.initial_conv(x)
        # block 1
        skip = self.conv11_1(x)
        x  = self.rb11_1(x)
        # block 2
        skip += self.conv11_4(x)
        x  = self.rb11_4(x)
        # block 3
        skip += self.conv21_10(x)
        x  = self.rb21_10(x)
        # final projections
        x = self.final_conv1(x) + skip
        x = self.final_conv2(x)
        # x = F.softmax(x, dim=1)  # since CrossEntropyLoss is used, we don't need to apply softmax here
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x
    
class SpliceAI_400nt(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=3):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1   = nn.Sequential(*[ResidualBlock(mid_channels, 11, 1) for _ in range(4)])
        self.conv11_4 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_4   = nn.Sequential(*[ResidualBlock(mid_channels, 11, 4) for _ in range(4)])
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)

        # Two convs per block, so each ResidualBlock adds 2*(kernel_size−1)*dilation.
        # Sum over all 4 blocks in each of the four groups:
        rf  = 1  # initial 1×1 conv has RF=1
        rf += 4 * 2 * (11 - 1) * 1    # first group
        rf += 4 * 2 * (11 - 1) * 4    # second group
        # total rf = 1 + 80 + 320 = 401
        self.receptive_field = rf
        # we want to drop rf//2 at each end -only predict the middle 5000 bases
        self.crop = (rf - 1) // 2      # = 200

    def forward(self, x):
        x = self.initial_conv(x)
        # block 1
        skip = self.conv11_1(x)
        x  = self.rb11_1(x)
        # block 2
        skip += self.conv11_4(x)
        x  = self.rb11_4(x)
        # final projections
        x = self.final_conv1(x) + skip
        x = self.final_conv2(x)
        # x = F.softmax(x, dim=1)  # since CrossEntropyLoss is used, we don't need to apply softmax here
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x

class SpliceAI_80nt(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, out_channels=3):
        super().__init__()

        # initial 1x1 bottleneck
        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        
        # single group of four 11-dilated residual blocks
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1 = nn.Sequential(*[ResidualBlock(mid_channels, 11, 1) for _ in range(4)])
        
        # final projections
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)

        # compute receptive field: 1 (initial) + 4 blocks * 2 convs * (11-1) * dilation=1
        rf = 1 + 4 * 2 * (11 - 1) * 1  # = 81
        self.receptive_field = rf
        # amount to crop from each end to only predict the central 80nt
        self.crop = (rf - 1) // 2  # = 40

    def forward(self, x):
        # x shape: (batch, in_channels, length)
        x = self.initial_conv(x)
        skip = self.conv11_1(x)
        x = self.rb11_1(x)
        x = self.final_conv1(x) + skip
        x = self.final_conv2(x)
        # x = F.softmax(x, dim=1) # since CrossEntropyLoss is used, we don't need to apply softmax here
        # crop flanking positions so output length = input_length - 2*crop
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop:-self.crop]
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_10k = SpliceAI_10k(in_channels=4, mid_channels=32, out_channels=3).to(device)
    model_2k  = SpliceAI_2k(in_channels=4, mid_channels=32, out_channels=3).to(device)
    model_400  = SpliceAI_400nt(in_channels=4, mid_channels=32, out_channels=3).to(device)
    model_80   = SpliceAI_80nt(in_channels=4, mid_channels=32, out_channels=3).to(device)
    # test the model with a short sequence
    x = torch.randn(2, 4, 5000+10_000) # (batch, in_channels, length)
    y = model_10k(x.to(device))
    print(y.shape)   # -> torch.Size([2, 3, 5000])
    x = torch.randn(2, 4, 5000+2000) # (batch, in_channels, length)
    y = model_2k(x.to(device))
    print(y.shape)   # -> torch.Size([2, 3, 5000])
    x = torch.randn(2, 4, 5000+400) # (batch, in_channels, length)
    y = model_400(x.to(device))
    print(y.shape)   # -> torch.Size([2, 3, 5000])
    x = torch.randn(2, 4, 5000+80) # (batch, in_channels, length)
    y = model_80(x.to(device))
    print(y.shape)   # -> torch.Size([2, 3, 5000])


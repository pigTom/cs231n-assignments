import torch
import torch.nn as nn
import numpy as np
from cs231n.classifiers.transformer import VisionTransformer

# Set random seeds
torch.manual_seed(231)
np.random.seed(231)

# Create a small dataset (one batch)
N, C, H, W = 32, 3, 32, 32
num_classes = 10
X = torch.randn(N, C, H, W)
y = torch.randint(0, num_classes, (N,))

print("=" * 60)
print("Testing ViT Overfit on One Batch")
print("=" * 60)

# Create model
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    in_channels=3,
    embed_dim=128,
    num_layers=6,
    num_heads=4,
    dim_feedforward=256,
    num_classes=10,
    dropout=0.0  # No dropout for overfitting
)

print(f"\nModel architecture:")
print(f"  - Patches: {(32//8)**2} patches of size 8x8")
print(f"  - Embed dim: 128")
print(f"  - Num layers: 6")
print(f"  - Num heads: 4")

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"  - Total parameters: {num_params:,}")

# Test forward pass
print("\n" + "-" * 60)
print("Testing forward pass...")
with torch.no_grad():
    output_test = model(X)
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape: {output_test.shape}")
    print(f"✓ Expected: ({N}, {num_classes})")
    assert output_test.shape == (N, num_classes), f"Shape mismatch!"

# Test backward pass
print("\n" + "-" * 60)
print("Testing backward pass...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

# Need to compute output WITH gradients for backward pass
output = model(X)
loss = criterion(output, y)
loss.backward()
print(f"✓ Loss computed: {loss.item():.4f}")
print(f"✓ Backward pass successful")

# Test if gradients are non-zero
has_grad = False
for name, param in model.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        has_grad = True
        break
print(f"✓ Gradients present: {has_grad}")

# Overfit test with different learning rates
print("\n" + "=" * 60)
print("Overfitting Test with Different Learning Rates")
print("=" * 60)

for lr in [1e-2, 5e-3, 1e-3]:
    print(f"\n--- Learning Rate: {lr} ---")

    # Reset model
    model = VisionTransformer(
        img_size=32, patch_size=8, in_channels=3,
        embed_dim=128, num_layers=6, num_heads=4,
        dim_feedforward=256, num_classes=10, dropout=0.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    losses = []
    accs = []

    for step in range(200):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = output.argmax(dim=1)
            acc = (pred == y).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if step % 50 == 0 or step == 199:
            print(f"[{step:3d}/200] Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    final_acc = accs[-1]
    if final_acc > 0.95:
        print(f"✓ SUCCESS! Final accuracy: {final_acc:.4f}")
        break
    else:
        print(f"✗ FAILED. Final accuracy: {final_acc:.4f}")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)

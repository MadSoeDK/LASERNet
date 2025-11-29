import torch
from loading import SliceSequenceDataset

dataset = SliceSequenceDataset(
    field="temperature",
    plane="xz",
    split="train",
    sequence_length=4,
    target_offset=1,
    preload=True
)

print(f"Scanning {len(dataset)} samples for min/max...")

global_min = float("inf")
global_max = float("-inf")

for sample in dataset:
    ctx = sample["context"]  # [seq_len, 1, H, W]
    tgt = sample["target"]   # [1, H, W]

    # Combine context + target
    all_frames = torch.cat([ctx, tgt.unsqueeze(0)], dim=0)

    frame_min = all_frames.min().item()
    frame_max = all_frames.max().item()

    global_min = min(global_min, frame_min)
    global_max = max(global_max, frame_max)

print("==== RESULTS ====")
print("Temperature MIN:", global_min)
print("Temperature MAX:", global_max)

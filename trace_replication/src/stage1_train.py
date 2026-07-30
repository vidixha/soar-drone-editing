"""
Stage 1 training-loop harness (setup only -- not run here, no data downloaded).

Expects a directory of precomputed .pt files, each holding one training example
already assembled by the (not-yet-built) pipeline that chains
stage1_source_filter.py -> recam_wrapper.py -> cotracker_wrapper.py into:
  {
    "first_frame_feat": (768,) float tensor,   # placeholder vision-encoder feature
    "point_tracks": (T, grid*grid, 2) float tensor,
    "ref_boxes": (T, 4) float tensor,           # B_ref, interpolated first-frame boxes
    "target_boxes": (T, 4) float tensor,        # ground truth box trajectory in the
                                                 # ReCamMaster-rendered target view
  }
Building that assembly script is the next piece of setup once GOT-10k is
actually downloaded; this harness only needs the .pt schema above to exist.

Paper doesn't give Stage 1's optimizer/batch-size/step-count (those are only
given for Stage 2); defaults below are ours, clearly not reported values.
"""
import argparse
import pathlib

import torch
from torch.utils.data import DataLoader, Dataset

from stage1_dit import CrossViewMotionDiT, flow_matching_loss


class Stage1PairDataset(Dataset):
    def __init__(self, data_dir: pathlib.Path):
        self.paths = sorted(data_dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(
                f"no .pt training pairs found under {data_dir} -- this harness "
                "expects data already assembled by the GOT-10k/ReCamMaster/"
                "CoTracker pipeline, see module docstring for the schema."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx])


def train(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    num_steps: int = 20000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    log_every: int = 50,
    checkpoint_every: int = 1000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Stage1PairDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CrossViewMotionDiT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break
            loss = flow_matching_loss(
                model,
                x1_boxes=batch["target_boxes"].to(device),
                first_frame_feat=batch["first_frame_feat"].to(device),
                point_tracks=batch["point_tracks"].to(device),
                ref_boxes=batch["ref_boxes"].to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"step {step}/{num_steps}  loss {loss.item():.4f}")
            if step > 0 and step % checkpoint_every == 0:
                torch.save(model.state_dict(), out_dir / f"stage1_dit_step{step}.pt")
            step += 1

    torch.save(model.state_dict(), out_dir / "stage1_dit_final.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 (Cross-View Motion Transformation) training")
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("checkpoints/stage1"))
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.out_dir, args.num_steps, args.batch_size, args.lr, args.device)

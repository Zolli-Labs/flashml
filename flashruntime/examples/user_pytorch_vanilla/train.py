"""Plain torch DDP — NO flashruntime import. Proves the launcher operates
unmodified code: the helper in ../user_pytorch is optional sugar."""
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    torch.manual_seed(0)
    model = torch.nn.Linear(16, 2)
    if world_size > 1:
        model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(256, 16, generator=g), torch.randint(0, 2, (256,), generator=g)
    loss = torch.tensor(0.0)
    for _ in range(100):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        with open("metrics.json", "w") as f:
            json.dump({"final_loss": round(loss.item(), 6)}, f)
        print("final loss", loss.item())
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

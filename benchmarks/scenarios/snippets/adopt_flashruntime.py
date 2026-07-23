# The same loop, made fault-tolerant with flashruntime. The diff against
# adopt_vanilla.py IS the adoption cost. Source: examples/user_pytorch/train.py
import torch
from torch.utils.data import DataLoader, TensorDataset

import flashruntime.torch as ft

x = torch.randn(512, 16)
y = (x.sum(1) > 0).long()
loader = DataLoader(TensorDataset(x, y), batch_size=32)
model = torch.nn.Linear(16, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.05)

model, opt, loader = ft.prepare(model, opt, loader)
start = ft.start_step()
for step, (xb, yb) in enumerate(loader, start=start):
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    ft.checkpoint(model, opt, step=step, every=50)

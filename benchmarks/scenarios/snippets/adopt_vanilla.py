# Plain PyTorch training loop — no fault tolerance, no resume.
import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(512, 16)
y = (x.sum(1) > 0).long()
loader = DataLoader(TensorDataset(x, y), batch_size=32)
model = torch.nn.Linear(16, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.05)

for step, (xb, yb) in enumerate(loader):
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()

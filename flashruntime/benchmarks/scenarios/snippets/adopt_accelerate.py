# The same loop, made distributed with Hugging Face Accelerate. (Accelerate
# gives distribution; crash-resume is still manual save_state/load_state.)
# Source: https://huggingface.co/docs/accelerate/basic_tutorials/migration
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

accelerator = Accelerator()
x = torch.randn(512, 16)
y = (x.sum(1) > 0).long()
loader = DataLoader(TensorDataset(x, y), batch_size=32)
model = torch.nn.Linear(16, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.05)

model, opt, loader = accelerator.prepare(model, opt, loader)
for step, (xb, yb) in enumerate(loader):
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    opt.zero_grad()
    accelerator.backward(loss)
    opt.step()
accelerator.save_state("ckpt")

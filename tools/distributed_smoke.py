import os

import torch
import torch.distributed as dist


torch.cuda.set_device(0)

dist.init_process_group("nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

x = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(x)

print(
    f"rank={rank}/{world_size} "
    f"host={os.uname().nodename} "
    f"device={torch.cuda.get_device_name()} "
    f"result={x.item()}"
)

# Mini-SGLang also uses a Gloo CPU group.
cpu_group = dist.new_group(backend="gloo")

y = torch.tensor([rank + 1.0])
dist.all_reduce(y, group=cpu_group)

print(f"rank={rank}: gloo result={y.item()}")

dist.destroy_process_group()
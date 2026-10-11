# Width concat of tiny ROW_MAJOR DRAM tensors: one case per process so a hang shows as a timeout.
import sys

import torch
import ttnn

n, w, dim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
shape = (w,) if dim == 0 else (1, w)
xs = [torch.randn(shape).to(torch.bfloat16) for _ in range(n)]
ts = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for x in xs]
print(f"INFO concat n={n} w={w} rank={len(shape)} starting", flush=True)
out = ttnn.to_torch(ttnn.concat(ts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)).float()
ref = torch.cat(xs, dim=-1).float()
print(f"RESULT concat_tiny n={n} w={w} rank={len(shape)}: bad={int((out.reshape(ref.shape) != ref).sum())}/{ref.numel()}", flush=True)
ttnn.close_device(device)

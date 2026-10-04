"""#53927: softmax dim=-2 on the H-large factory under the watcher. One case per process: python probe53927.py <case>"""
import sys

import torch
import ttnn

CASES = {
    "h_large": ((1, 1, 4096, 32), -2),
    "h_small": ((1, 1, 512, 32), -2),
    "w_large": ((1, 1, 1, 32, 4096), -1),
}
shape, dim = CASES[sys.argv[1]]
torch.manual_seed(0)
x = torch.rand(shape).to(torch.bfloat16)
dev = ttnn.open_device(device_id=0)
try:
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    y = ttnn.to_torch(ttnn.softmax(t, dim=dim)).float()
    d = (y - torch.softmax(x.float(), dim)).abs().max().item()
    print(f"RESULT {sys.argv[1]} {list(shape)} dim={dim}: ran, max_abs_diff {d:.6f}")
except Exception as e:
    print(f"RESULT {sys.argv[1]} {list(shape)} dim={dim}: EXCEPTION {str(e)[:500]}")
finally:
    ttnn.close_device(dev)

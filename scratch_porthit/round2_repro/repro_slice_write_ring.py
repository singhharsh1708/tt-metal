# slice_write RM interleaved: ring sized from the aligned page, per-core reserve from the raw row.
import sys

import torch
import ttnn

Hn, Wn = (16385, 8) if sys.argv[1] == "bad" else (256, 128)
device = ttnn.open_device(device_id=0)
inp = (torch.arange(Hn * Wn) % 251).reshape(Hn, Wn).to(torch.uint8)
ti = ttnn.from_torch(inp, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint8)
to = ttnn.from_torch(torch.zeros(Hn, Wn, dtype=torch.uint8), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint8)
print(f"INFO slice_write [{Hn},{Wn}] uint8 starting (a hang shows as rc=124)", flush=True)
ttnn.experimental.slice_write(ti, to, [0, 0], [Hn, Wn], [1, 1])
res = ttnn.to_torch(to)
print(f"RESULT slice_write [{Hn},{Wn}] uint8: bad={int((res != inp).sum())}/{inp.numel()}", flush=True)
ttnn.close_device(device)

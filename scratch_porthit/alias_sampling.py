"""ttnn.sampling: after a call that passes ONE tensor as both p and temp, a later call with distinct p/temp hits the
same cache entry and reads them swapped (base spec path replays binding indices)."""
import torch
import ttnn

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
if hasattr(dev, "enable_program_cache"):
    dev.enable_program_cache()
U, W = 32, 64
vals = torch.randn(1, 1, U, W)
hot = torch.tensor([(7 * u + 3) % W for u in range(U)])
vals[0, 0, torch.arange(U), hot] = 12.0
idx = torch.arange(W, dtype=torch.int32).expand(1, 1, U, W).contiguous()
tv = ttnn.from_torch(vals, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ti = ttnn.from_torch(idx, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
rm = lambda t, dt: ttnn.from_torch(t, device=dev, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
k = rm(torch.tensor([32] * U), ttnn.uint32)
p_zero, t_one, shared = rm(torch.zeros(U), ttnn.bfloat16), rm(torch.ones(U), ttnn.bfloat16), rm(torch.ones(U), ttnn.bfloat16)
entries = lambda: dev.num_program_cache_entries() if hasattr(dev, "num_program_cache_entries") else -1


def run(p, temp, seed):
    return ttnn.to_torch(ttnn.sampling(tv, ti, k=k, p=p, temp=temp, seed=seed)).reshape(-1)[:U].to(torch.int64)


def report(tag, got):
    n = int((got == hot).sum())
    print(f"RESULT alias_sampling {tag}: matches_argmax={n}/{U} {'OK' if n == U else 'WRONG'}", flush=True)


try:
    report("control_distinct_first_seed1111", run(p_zero, t_one, 1111))
    e0 = entries(); run(shared, shared, 42); e1 = entries()
    b = run(p_zero, t_one, 42); e2 = entries()
    print(f"INFO cache entries: before A {e0}, after A {e1}, after B {e2} (B unchanged = cache hit)", flush=True)
    report("B_after_aliased_A_seed42", b)
    dev.disable_and_clear_program_cache()
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    report("control_B_after_cache_clear_seed42", run(p_zero, t_one, 42))
except Exception as e:
    print(f"RESULT alias_sampling EXC {type(e).__name__}: {str(e).strip().splitlines()[0][:240]}", flush=True)
ttnn.close_device(dev)

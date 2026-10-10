"""ttnn.manual_seed: after a call with ONE tensor as both seeds and user_ids, a later distinct call is bound swapped."""
import torch
import ttnn

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
if hasattr(dev, "enable_program_cache"):
    dev.enable_program_cache()
U, W = 32, 64
tv = ttnn.from_torch(torch.randn(1, 1, U, W), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ti = ttnn.from_torch(torch.arange(W, dtype=torch.int32).expand(1, 1, U, W).contiguous(), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
rm = lambda t, dt: ttnn.from_torch(t, device=dev, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
k, p_one, t_one = rm(torch.tensor([32] * U), ttnn.uint32), rm(torch.ones(U), ttnn.bfloat16), rm(torch.ones(U), ttnn.bfloat16)
perm = [(5 * i + 3) % U for i in range(U)]
inv = [0] * U
for i, v in enumerate(perm):
    inv[v] = i
user_ids, seeds, seeds_inv, shared = (rm(torch.arange(U), ttnn.uint32), rm(torch.tensor(perm), ttnn.uint32),
                                      rm(torch.tensor(inv), ttnn.uint32), rm(torch.arange(U), ttnn.uint32))
draw = lambda: ttnn.to_torch(ttnn.sampling(tv, ti, k=k, p=p_one, temp=t_one)).reshape(-1)[:U].to(torch.int64)


def seeded(s, u):
    ttnn.manual_seed(seeds=s, user_ids=u)
    return draw()


def clear():
    dev.disable_and_clear_program_cache()
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()


same = lambda a, b: bool(torch.equal(a, b))
try:
    ref = seeded(seeds, user_ids); again = seeded(seeds, user_ids); swapped = seeded(seeds_inv, user_ids)
    print(f"RESULT alias_seed control_determinism same={same(ref, again)} (must be True)", flush=True)
    print(f"RESULT alias_seed control_sensitivity inverse_differs={not same(ref, swapped)} (must be True)", flush=True)
    clear(); ttnn.manual_seed(seeds=shared, user_ids=shared)
    test = seeded(seeds, user_ids)
    print(f"RESULT alias_seed B_after_aliased_A equals_reference={same(test, ref)} equals_swapped={same(test, swapped)} "
          f"{'OK' if same(test, ref) else 'WRONG'}", flush=True)
    clear(); ctl = seeded(seeds, user_ids)
    print(f"RESULT alias_seed control_B_after_cache_clear equals_reference={same(ctl, ref)}", flush=True)
except Exception as e:
    print(f"RESULT alias_seed EXC {type(e).__name__}: {str(e).strip().splitlines()[0][:240]}", flush=True)
ttnn.close_device(dev)

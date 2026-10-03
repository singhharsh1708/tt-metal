# Build current tt-metal main on the console in its own tree (~/ttm-main), next to the stable /home/user/tt-metal.
# Configure-only gate first; full build only if configure passes. Then a device smoke test.
# Afterwards `source ~/ttm-main-env.sh` switches a shell to the main tree.
set -uo pipefail
unset LD_LIBRARY_PATH
D=/home/user/ttm-main
LOG=/home/user/ttm-main-logs; mkdir -p "$LOG"
echo "cpus $(nproc)  disk free $(df -h /home/user | tail -1 | awk '{print $4}')  python $(python -c 'import sys;print(sys.version.split()[0])')  clang $(clang++-20 --version 2>/dev/null | head -1 || echo none)"
FREE_GB=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
if [ "$FREE_GB" -lt 60 ]; then echo "only ${FREE_GB}G free; a second tree plus build needs ~60G. Free space first."; exit 1; fi

if [ ! -d "$D/.git" ]; then git clone -q --filter=blob:none https://github.com/tenstorrent/tt-metal.git "$D" || exit 1; fi
cd "$D" || exit 1
git fetch -q origin main && git checkout -q --detach 4502c6d9c57 || exit 1
git submodule update --init --recursive -q || { echo "submodule update failed"; exit 1; }
echo "main at $(git log -1 --format='%h %cd %s' --date=short | cut -c1-90)"

export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn
cat > /home/user/ttm-main-env.sh <<ENV
unset LD_LIBRARY_PATH
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn
cd $D
ENV

echo "=== configure (gate) ==="
t0=$(date +%s)
./build_metal.sh --disable-profiler --configure-only > "$LOG/configure.log" 2>&1; rc=$?
echo "configure rc=$rc in $(( $(date +%s) - t0 ))s"
if [ $rc -ne 0 ]; then echo "--- configure failed, last lines:"; grep -vE "^\s*$" "$LOG/configure.log" | tail -25 | cut -c1-220; exit 1; fi
grep -E "Using SFPI compiler|CMAKE_CXX_COMPILER|Build files have been written" "$LOG/configure.log" | head -4 | cut -c1-200

echo "=== build ==="
t0=$(date +%s)
./build_metal.sh --disable-profiler > "$LOG/build.log" 2>&1; rc=$?
echo "build rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"
if [ $rc -ne 0 ]; then echo "--- build failed, first errors:"; grep -nE "error:|Error [0-9]|FAILED:" "$LOG/build.log" | head -15 | cut -c1-220; exit 1; fi
ls -la build/lib/_ttnncpp.so ttnn/ttnn/_ttnn*.so 2>/dev/null | awk '{print $5, $9}'

echo "=== device smoke test on main ==="
rm -rf /home/user/.cache/tt-metal-cache
python - <<'PY'
import torch, ttnn
print("ttnn from", ttnn.__file__)
dev = ttnn.open_device(device_id=0)
x = torch.randn(64, 64)
y = ttnn.to_torch(ttnn.relu(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))).float()
print("relu max err", (y - x.clamp(min=0).bfloat16().float()).abs().max().item())
q = torch.randn(1, 1, 8, 64); k = torch.randn(1, 2, 128, 64); v = torch.randn(1, 2, 128, 64)
o = ttnn.transformer.scaled_dot_product_attention_decode(
    ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), cur_pos=[100], scale=0.125)
print("sdpa decode ran, shape", tuple(ttnn.to_torch(o).shape))
ttnn.close_device(dev)
print("SMOKE OK")
PY
echo "done: main tree at $D; use: source /home/user/ttm-main-env.sh"

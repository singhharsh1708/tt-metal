# Build CURRENT tt-metal main, unpatched, in /home/user/ttm-latest, then re-run the model-sweep leads there.
# STAGE=build | run | all (default all). Re-running with STAGE=run skips the build.
set -uo pipefail
unset LD_LIBRARY_PATH CI
STAGE=${STAGE:-all}
D=/home/user/ttm-latest
LOG=/home/user/ttm-latest-logs; mkdir -p "$LOG"
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
if [ "$STAGE" != run ]; then
  FREE_GB=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE_GB" -lt 80 ] && { echo "only ${FREE_GB}G free, need 80G"; exit 1; }
  if [ ! -d "$D/.git" ]; then git clone -q --filter=blob:none https://github.com/tenstorrent/tt-metal.git "$D" || exit 1; fi
  cd "$D" || exit 1
  git fetch -q origin main && git checkout -q --detach origin/main || exit 1
  git submodule sync -q --recursive
  if ! git submodule update --init --recursive -q; then
    git submodule deinit -f --all -q; rm -rf .git/modules
    git submodule update --init --recursive -q || { echo "submodule update failed"; exit 1; }
  fi
  echo "latest main at $(git log -1 --format='%H %cd %s' --date=short | cut -c1-110)"
  export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn
  t0=$(date +%s)
  ./build_metal.sh --disable-profiler > "$LOG/build.log" 2>&1; rc=$?
  echo "build rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"
  if [ $rc -ne 0 ]; then grep -nE "error:|Error [0-9]|FAILED:" "$LOG/build.log" | head -15 | cut -c1-220; tail -5 "$LOG/build.log" | cut -c1-220; exit 1; fi
  rm -rf /home/user/.cache/tt-metal-cache
  python - <<'PY' 2>&1 | tail -4
import torch, ttnn
print("ttnn from", ttnn.__file__)
dev = ttnn.open_device(device_id=0)
x = torch.randn(64, 64)
y = ttnn.to_torch(ttnn.relu(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))).float()
print("relu max err", (y - x.clamp(min=0).bfloat16().float()).abs().max().item())
ttnn.close_device(dev)
print("SMOKE OK")
PY
fi
[ "$STAGE" = build ] && exit 0
cd "$D" || { echo "no $D, run the build stage first"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
echo "tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
RUN=/home/user/leads_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
NT="-p no:timeout"
pt () {  # label, timeout_s, pytest args...
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|skipped).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-140)"
  grep -aE "^(FAILED|ERROR) |^E +(AssertionError|RuntimeError|ValueError|[A-Za-z]*Error)|TT_THROW:|TT_FATAL:|Top1 Accuracy|test_model_inference:[0-9]+ - PCC|PCC value|PCC=|centralized|Statically allocated" "$RUN/$1.log" | grep -av "riscv-tt-elf" | awk '!s[$0]++' | tail -14 | cut -c1-250
}
out () {  # label: generated text samples
  grep -aE "\[User 0\]" "$RUN/$1.log" | awk 'NR<=2; {l=$0} END{print l}' | cut -c1-260
}
TTT=models/tt_transformers; WH=models/demos/wormhole
SD=models/demos/vision/generative/stable_diffusion/wormhole
pt bge_large_pcc 900 --timeout=840 $WH/bge_large_en/tests/pcc/test_ttnn_bge_model.py
pt bge_large_e2e 900 --timeout=840 "$WH/bge_large_en/tests/perf/test_bge_e2e_performant.py::test_e2e_performant_bge"
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150
pt phi3_perf_tokmatch 1500 --timeout=1440 $TTT/demo/simple_text_demo.py -k "performance and ci-token-matching"
pt phi3_acc_tokmatch 1500 --timeout=1440 $TTT/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
pt phi3_perf_ci1 900 --timeout=840 $TTT/demo/simple_text_demo.py -k "performance and ci-1"; out phi3_perf_ci1
pt phi3_long16k 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "performance and long-context-16k and not ci-long"; out phi3_long16k
export HF_MODEL=Qwen/Qwen2.5-3B-Instruct
pt qwen25_3b_model_pcc 1800 $NT $TTT/tests/test_model.py -k "full"
unset HF_MODEL MESH_DEVICE
pt sd14_unet 1500 $NT $SD/tests/test_unet_2d_condition_model.py
echo "done $(date -u +%H:%M)  logs: $RUN"

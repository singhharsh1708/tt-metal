# Deciding runs for BGE-large, Qwen2.5-3B and SD 1.4 on the unpatched latest-main tree. Restores the tree at the end.
set -uo pipefail
unset LD_LIBRARY_PATH CI
D=/home/user/ttm-latest
cd "$D" || { echo "no $D"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
RUN=/home/user/three_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
pt () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider -rA "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|skipped|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-140)"
  grep -aE "^(FAILED|PASSED|ERROR) |^E +[A-Za-z]*Error|TT_THROW:|TT_FATAL:| PCC: |PCC value|PCC=|pcc = |requires specific TensorPrecision|performant_runner.py:[0-9]+" "$RUN/$1.log" | grep -av "riscv-tt-elf" | awk '!s[$0]++' | tail -${N:-14} | cut -c1-230
}
BT=models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_model.py
# 1. BGE-large: the passing PCC test with the e2e test's seed
grep -c "random_seed = 42" $BT
sed -i 's/random_seed = 42/random_seed = 0/' $BT
pt bge_pcc_seed0 900 --timeout=840 $BT -s
git checkout -q -- $BT
# 2. Qwen2.5-3B performance mode with the MLP at BFP8 + HIFI2_FP16 instead of BFP4 + LoFi
export HF_MODEL=Qwen/Qwen2.5-3B-Instruct MESH_DEVICE=N150
J=models/tt_transformers/model_params/Qwen2.5-3B-Instruct; mkdir -p $J
python3 -c "import json; json.dump({'decoders': {str(i): {'precision_cfg': {'FF1_FF3': 'BFP8'}, 'fidelity_cfg': {'LI_FF1_FF3': 'HIFI2_FP16'}} for i in range(36)}}, open('$J/performance_decoder_config.json','w'), indent=1)"
pt qwen3b_perf_mlp_bfp8 1800 -p no:timeout models/tt_transformers/tests/test_model.py -k "full and performance"
rm -rf $J
unset HF_MODEL MESH_DEVICE
# 3. SD 1.4: every sub-module test (all passed in CI on 2026-08-10)
N=40 pt sd14_modules 3000 -p no:timeout models/demos/vision/generative/stable_diffusion/wormhole/tests -k "not vae and not perf"
echo "tree restored, dirty files: $(git status --porcelain | grep -vc '^??') untracked in model_params: $(git status --porcelain models/tt_transformers/model_params | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"

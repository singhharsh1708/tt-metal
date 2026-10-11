# Does #56292 also move BGE-large? And does restoring the legacy rsqrt flag in SD 1.4's layer norm recover the UNet on main?
set -uo pipefail
unset LD_LIBRARY_PATH CI
D=/home/user/ttm-latest
cd "$D" || { echo "no $D"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/f56292_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
MAIN=$(git rev-parse HEAD)
echo "start tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
pt () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider -p no:timeout -rA "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-90) | fail value: $(grep -aE '^E +AssertionError: [0-9.]+' "$RUN/$1.log" | awk '{print $3}' | tr '\n' ' ') $(grep -aE '^E +[A-Za-z]*Error' "$RUN/$1.log" | grep -av AssertionError | head -1 | cut -c1-140)"
}
SD=models/demos/vision/generative/stable_diffusion/wormhole
BT=models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_model.py
# 1. SD 1.4 on main with the legacy rsqrt flag restored in the transformer block's layer norm
F=$SD/tt/ttnn_functional_basic_transformer_block.py
grep -n "LayerNormDefaultProgramConfig" $F | cut -c1-160
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "LayerNormDefaultProgramConfig(legacy_reduction=True)"
print("occurrences", s.count(old))
if s.count(old) >= 1:
    open(p, "w").write(s.replace(old, "LayerNormDefaultProgramConfig(legacy_reduction=True, legacy_rsqrt=True)"))
PY
pt main_legacy_ln_downblock 900 $SD/tests/test_cross_attn_downblock_2d.py
pt main_legacy_ln_unet 1500 $SD/tests/test_unet_2d_condition_model.py
git checkout -q -- $F
# 2. BGE-large seeds 0 and 2 on the commit before #56292 and on #56292
bge () {
  for s in 0 2; do
    [ -f $BT ] || { echo "no $BT at this commit"; return; }
    sed -i "s/random_seed = 42/random_seed = $s/" $BT
    pt "$1_bge_seed$s" 600 $BT -k inputs0 -s
    git checkout -q -- $BT
  done
}
build () {
  git checkout -q --detach "$1" || return 1
  git submodule update --init --recursive -q || return 1
  t0=$(date +%s); ./build_metal.sh --disable-profiler > "$RUN/build_$2.log" 2>&1; rc=$?
  echo "=== build [$2 $(git log -1 --format='%h %cd' --date=short)] rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"; return $rc
}
build af556315cea before_56292 && bge before_56292
build 846c8f5ba23b at_56292 && bge at_56292
build "$MAIN" main_restored
echo "end tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??').  done $(date -u +%H:%M)  logs: $RUN"

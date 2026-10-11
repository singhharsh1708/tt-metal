# BGE-large seed spread on clean main, then SD 1.4: does #56292 (846c8f5ba23b) move the UNet PCC? Ends back on main.
set -uo pipefail
unset LD_LIBRARY_PATH CI
D=/home/user/ttm-latest
cd "$D" || { echo "no $D"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/sdbisect_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
MAIN=$(git rev-parse HEAD)
echo "start tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
pt () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider -p no:timeout -rA "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-110) | $(grep -aE '^E +AssertionError: [0-9.]+' "$RUN/$1.log" | awk '{print $3}' | tr '\n' ' ') $(grep -aoE 'PCC[ =:]+0\.[0-9]+|pcc[ =:]+0\.[0-9]+|[Pp]cc.{0,20}0\.9[0-9]+' "$RUN/$1.log" | sort -u | tail -4 | tr '\n' ' ')"
}
BT=models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_model.py
for s in 1 2 3 4 5 6; do
  sed -i "s/random_seed = 42/random_seed = $s/" $BT
  pt bge_seed$s 600 $BT -k inputs0 -s
  git checkout -q -- $BT
done
SD=models/demos/vision/generative/stable_diffusion/wormhole/tests
sd () { pt "$1_downblock" 900 $SD/test_cross_attn_downblock_2d.py; pt "$1_unet" 1500 $SD/test_unet_2d_condition_model.py; }
build () {
  git checkout -q --detach "$1" || return 1
  git submodule update --init --recursive -q || return 1
  t0=$(date +%s); ./build_metal.sh --disable-profiler > "$RUN/build_$2.log" 2>&1; rc=$?
  echo "=== build [$2 $(git log -1 --format='%h %cd' --date=short)] rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"; return $rc
}
sd main
if build af556315cea before_56292; then sd before_56292; else grep -aE "error:|FAILED:" "$RUN/build_before_56292.log" | head -5 | cut -c1-200; fi
if build 846c8f5ba23b at_56292; then sd at_56292; else grep -aE "error:|FAILED:" "$RUN/build_at_56292.log" | head -5 | cut -c1-200; fi
build "$MAIN" main_restored
echo "end tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??').  done $(date -u +%H:%M)  logs: $RUN"

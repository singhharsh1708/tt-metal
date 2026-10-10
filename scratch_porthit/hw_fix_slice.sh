set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
F=ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
RUN=/home/user/fixslice_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- "$F"
echo "tree: $(git log -1 --format='%h %cd' --date=short)  artifacts: $RUN"
python - "$F" <<'PY' || { echo "PATCH FAILED, nothing built"; exit 1; }
import sys
p = sys.argv[1]; s = open(p).read()
a = "    const uint32_t stick_size_aligned = tt::round_up(unpadded_row_size_bytes, alignment);\n"
b = "    const uint32_t stick_size_aligned = tt::round_up(unpadded_row_size_bytes + misalignment, alignment);\n"
c = "                auto num_sticks_per_core_pad32 = round_up_to_mul32(num_sticks_per_core);\n"
d = ("                auto num_sticks_per_core_pad32 =\n"
     "                    round_up_to_mul32(std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2));\n")
assert s.count(a) == 1, ("entry size anchor", s.count(a))
assert s.count(c) == 1, ("per-core batch anchor", s.count(c))
open(p, "w").write(s.replace(a, b).replace(c, d))
PY
git diff -- "$F" | grep -E "^[+-] "
echo "=== rebuild start $(date -u +%H:%M)"
t0=$(date +%s)
if [ -f build/build.ninja ]; then ninja -C build > "$RUN/build.log" 2>&1; else cmake --build build -- -j"$(nproc)" > "$RUN/build.log" 2>&1; fi
RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"; [ $RC -ne 0 ] && { grep -E "error|FAILED" "$RUN/build.log" | head -12 | cut -c1-240; echo "BUILD FAILED; source left patched, see $RUN/build.log"; exit 1; }
one () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 900 "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"; grep -E '^RESULT' "$RUN/$1.log" | cut -c1-260; }
one slice_ring_patched python ~/slice_ring.py
one slice_57382_patched python ~/cand_probe.py slice
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
T=tests/ttnn/unit_tests/operations/data_movement/test_slice.py
timeout -k 20 2400 pytest -q -p no:cacheprovider "$T" > "$RUN/test_slice.log" 2>&1; echo "===== [TT test_slice.py patched] rc=$? | $(tail -1 "$RUN/test_slice.log" | cut -c1-140)"
grep -E "^(FAILED|ERROR) " "$RUN/test_slice.log" | head -8 | cut -c1-200
echo "source is LEFT PATCHED and built (dirty: $(git status --porcelain | wc -l | tr -d ' ')).  done $(date -u +%H:%M)"

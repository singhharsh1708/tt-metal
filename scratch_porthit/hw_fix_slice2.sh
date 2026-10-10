set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
F=ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
RUN=/home/user/fixslice2_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
echo "patch present in source: $(grep -c 'unpadded_row_size_bytes + misalignment' "$F") (want 1)"
echo "--- which library python loads"
python - <<'PY'
import ttnn, os, time
for name in ("_ttnn", "_ttnncpp"):
    m = getattr(ttnn, name, None)
    print(name, getattr(m, "__file__", None))
print("ttnn package:", ttnn.__file__)
pid = os.getpid()
seen = set()
for line in open(f"/proc/{pid}/maps"):
    p = line.strip().split()[-1]
    if ("ttnn" in p or "tt_metal" in p) and p.endswith(".so") and p not in seen:
        seen.add(p)
        print("LOADED", p, time.strftime("%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(p))), "->", os.path.realpath(p))
PY
echo "--- libraries on disk (UTC mtime)"
find build ttnn/ttnn -maxdepth 3 \( -name "_ttnn*.so" -o -name "libtt_metal.so" \) -printf "%TY-%Tm-%Td %TH:%TM  %p\n" 2>/dev/null | sort | tail -12
echo "=== install start $(date -u +%H:%M)"
t0=$(date +%s); cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$? $(( $(date +%s) - t0 ))s"
find build ttnn/ttnn -maxdepth 3 -name "_ttnn*.so" -printf "%TY-%Tm-%Td %TH:%TM  %p\n" 2>/dev/null | sort | tail -6
one () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 900 "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"; grep -E '^RESULT' "$RUN/$1.log" | cut -c1-200; }
one slice_ring_patched python ~/slice_ring.py
one slice_57382_patched python ~/cand_probe.py slice
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 2400 pytest -q -p no:cacheprovider tests/ttnn/unit_tests/operations/data_movement/test_slice.py > "$RUN/test_slice.log" 2>&1
echo "===== [TT test_slice.py patched] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/test_slice.log" | tail -1 | cut -c1-140)"
grep -E "^(FAILED|ERROR) " "$RUN/test_slice.log" | head -8 | cut -c1-200
echo "done $(date -u +%H:%M)  logs: $RUN"

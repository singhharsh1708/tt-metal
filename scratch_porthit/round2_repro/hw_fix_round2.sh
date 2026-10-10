set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
R=${R2:-$HOME/round2_repro}
O=ttnn/cpp/ttnn/operations
DIRS="$O/data_movement/clone $O/data_movement/transpose $O/pool/rotate $O/pool/generic $O/experimental/cnn/convert_to_hwc $O/data_movement/indexed_fill $O/experimental/paged_cache $O/transformer/sdpa_decode $O/transformer/sdpa $O/moreh/moreh_getitem $O/experimental/padded_slice $O/experimental/slice_write $O/data_movement/non_zero_indices"
RUN=/home/user/fixround2_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- $DIRS
python $R/patch_rowpitch.py && python $R/patch_aux_misc.py || { echo "PATCH FAILED, nothing built"; git checkout -q -- $DIRS; exit 1; }
git diff --stat -- $DIRS | tail -1
echo "=== rebuild start $(date -u +%H:%M)"; t0=$(date +%s)
ninja -C build > "$RUN/build.log" 2>&1; RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED" "$RUN/build.log" | head -20 | cut -c1-300; echo "BUILD FAILED; source LEFT PATCHED for diagnosis, nothing installed. log: $RUN/build.log"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
one () {  # label, timeout, command...
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 "$2" "${@:3}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"
  grep -E '^ *(RESULT|INFO)|^(float32|bfloat16|int32)' "$RUN/$1.log" | cut -c1-330
  grep -E "error:|TT_FATAL|TT_THROW|Traceback" "$RUN/$1.log" | tail -2 | cut -c1-260
}
for c in clone transpose_hc rotate pool to_hwc indexed_fill; do one rp_$c 600 python $R/repro_rowpitch_family.py $c; done
one fill_cache_pt 600 python $R/repro_a_paged_fill_cache_tile_page_table.py
one sdpa_cur_pos 600 python $R/repro_d_sdpa_decode_tile_cur_pos.py
one ring_sdpa_pt 900 python $R/repro_c_ring_sdpa_tile_page_table.py
one getitem_mixed 600 python $R/repro_b_moreh_getitem_mixed_index_layouts.py
one nonzero_bf8 600 python $R/repro_f_nonzero_bfloat8.py
one padded_slice_fp32 600 python $R/repro_padded_slice_fp32.py
one slice_write_ctl 300 python $R/repro_slice_write_ring.py ctl
one slice_write_bad 300 python $R/repro_slice_write_ring.py bad
pt () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 2400 pytest -q -p no:cacheprovider "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"; grep -E "^(FAILED|ERROR) " "$RUN/$1.log" | head -6 | cut -c1-200; }
for n in test_clone test_transpose test_permute test_rotate test_maxpool2d test_avgpool2d test_convert_to_hwc test_indexed_fill test_paged_fill_cache test_paged_update_cache test_sdpa_decode test_sdpa_ring_distributed test_moreh_getitem test_padded_slice test_slice_write test_non_zero_indices test_slice_for_conv; do
  f=$(find tests/ttnn/unit_tests tests/ttnn/nightly/unit_tests tests/tt_eager/python_api_testing/unit_testing -name "$n.py" 2>/dev/null | head -1)
  if [ -n "$f" ]; then pt "tt_$n" "$f"; else echo "===== [tt_$n] no such test file"; fi
done
echo "source LEFT PATCHED, dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"

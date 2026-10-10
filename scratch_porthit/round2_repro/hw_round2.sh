set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: rebuild first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)"
R=${R2:-$HOME/round2_repro}
RUN=/home/user/round2_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
one () {  # label, timeout, command...
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 "$2" "${@:3}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"
  grep -E '^ *(RESULT|INFO|nlp_|ttnn\.|create_|  [QKV]:)|^(float32|bfloat16|int32)' "$RUN/$1.log" | cut -c1-330
  grep -E "TT_FATAL|TT_THROW|Traceback" "$RUN/$1.log" | tail -2 | cut -c1-240
}
for c in clone transpose_hc concat2 concat3 rotate pool to_hwc indexed_fill; do one rp_$c 600 python $R/repro_rowpitch_family.py $c; done
one fill_cache_pt 600 python $R/repro_a_paged_fill_cache_tile_page_table.py
one sdpa_cur_pos 600 python $R/repro_d_sdpa_decode_tile_cur_pos.py
one ring_sdpa_pt 900 python $R/repro_c_ring_sdpa_tile_page_table.py
one getitem_mixed 600 python $R/repro_b_moreh_getitem_mixed_index_layouts.py
one nonzero_bf8 600 python $R/repro_f_nonzero_bfloat8.py
one scatter_guard 600 python $R/probe_scatter_int32_tile_guard.py
one padded_slice_fp32 600 python $R/repro_padded_slice_fp32.py
one qkv_k_tf32 600 python $R/repro_qkv_k_tf32.py
one slice_write_ctl 300 python $R/repro_slice_write_ring.py ctl
one slice_write_bad 300 python $R/repro_slice_write_ring.py bad
echo "done $(date -u +%H:%M)  logs: $RUN"

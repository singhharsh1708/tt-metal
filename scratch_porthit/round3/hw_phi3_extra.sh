# Phi-3-mini-128k deciding runs on the unpatched latest-main tree (run after hw_leads_latest.sh).
set -uo pipefail
unset LD_LIBRARY_PATH CI
D=/home/user/ttm-latest
cd "$D" || { echo "no $D, run hw_leads_latest.sh first"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
RUN=/home/user/phi3x_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
pt () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|skipped|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-140)"
  grep -aE "^(FAILED|ERROR) |^E +[A-Za-z]*Error|TT_THROW:|TT_FATAL:|Top1 Accuracy|Accuracy Check Failed|lower than expected|Encoded prompt lengths|Prefill seq len: [0-9]{4,}" "$RUN/$1.log" | grep -av "riscv-tt-elf" | awk '!s[$0]++' | tail -10 | cut -c1-230
  grep -aE "\[User 0\]" "$RUN/$1.log" | awk 'NR<=2; {l=$0} END{print l}' | cut -c1-260
}
TTT=models/tt_transformers
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150
# 1. same 16k prompt, prefill in 4k chunks instead of one 32k shot
MAX_PREFILL_CHUNK_SIZE=4 pt phi3_long16k_chunk4 2400 --timeout=2340 $TTT/demo/simple_text_demo.py -k "performance and long-context-16k and not ci-long"
# 2. first failing length: 32 tokens generated at each context length
pt phi3_seqlen_sweep 3000 -p no:timeout $TTT/demo/simple_text_demo.py -k "performance and seqlen-sweep"
# 3. is long context broken for a CI-covered model too (public weights only)
export HF_MODEL=Qwen/Qwen2.5-3B-Instruct
pt qwen3b_long16k 2400 --timeout=2340 $TTT/demo/simple_text_demo.py -k "performance and long-context-16k and not ci-long"
echo "done $(date -u +%H:%M)  logs: $RUN"

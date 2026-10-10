#!/usr/bin/env bash
# Shipped-model sweep for one Wormhole N150, tt-metal @ 4502c6d9c57.
# Default run: groups A, B, C (public weights, target under 3 h).
# RUN_BIG=1 adds group D (11 to 15 GB weights each). RUN_GATED=1 adds group E (HF token needed).
# CLEAN=1 deletes each model's HF cache and tt cache after its lines (frees disk).
set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/modelsweep_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
df -h /home/user | tail -1
pt () {  # label, timeout_s, pytest args...
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -E '(passed|failed|error|skipped).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-140)"
  grep -E "^(FAILED|ERROR) |PCC|pcc|accuracy|top-1|Top-1|TT_FATAL|TT_THROW|OSError|gated|401|403" "$RUN/$1.log" | tail -8 | cut -c1-240
}

# ---- additions to the skeleton -------------------------------------------------
export HF_HUB_OFFLINE=0
RUN_BIG=${RUN_BIG:-0}; RUN_GATED=${RUN_GATED:-0}; CLEAN=${CLEAN:-0}
T0=$(date +%s)
# pytest.ini sets timeout=300 per test. NT disables pytest-timeout so the outer
# `timeout` is the only bound (needed where a first-run download or a 600 s
# marker would kill the test early). Lines without NT pass --timeout instead.
NT="-p no:timeout"
mx () {  # label, regex: extra metric lines the pt grep does not catch (case-insensitive)
  grep -iE "$2" "$RUN/$1.log" 2>/dev/null | tail -6 | cut -c1-240
}
need () {  # min free GB on /home/user, else skip the next pt
  local f; f=$(df -BG --output=avail /home/user | tail -1 | tr -dc '0-9')
  if [ "${f:-0}" -lt "$1" ]; then echo "===== SKIP next line: ${f} GB free, need $1 GB"; return 1; fi
}
hfrm () {  # CLEAN=1 only: drop HF hub cache and tt cache for org/name
  [ "$CLEAN" = 1 ] || return 0
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}" "$TT_METAL_HOME/model_cache/$1"
}
el () { echo "----- elapsed $(( ($(date +%s) - T0) / 60 )) min | $(df -h /home/user | tail -1 | awk '{print $4" free"}')"; }

TTT=models/tt_transformers
SEG=models/demos/vision/segmentation
CLS=models/demos/vision/classification
SD=models/demos/vision/generative/stable_diffusion/wormhole
WH=models/demos/wormhole
EXP=models/experimental
WSP=models/demos/audio/whisper

# =============================================================================
# GROUP A. README-listed N150 models removed from every CI pipeline on
# 2026-08-10 (PR #52318) and never re-added: SegFormer, MobileNetV2,
# SentenceBERT, Vanilla-UNet, VGG-UNet, UFLD v2, SD 1.4, Swin-S, Swin-V2,
# EfficientNet-B0, VoVNet. No run since, across all Metal 2.0 op ports.
# =============================================================================

# SegFormer (models/README: 512x512, N150, 132 FPS). Uses nlp_create_qkv_heads_segformer,
# ported to ProgramDescriptor on 2026-09-25 (#57409). Not in any CI job.
# Thresholds: semantic seg logits PCC 0.979, classification 0.96, encoder model 0.91.
# Weights: nvidia/segformer-b0-finetuned-ade-512-512 + nvidia/mit-b0, public HF, ~40 MB.
pt segformer_pcc_full 600 --timeout=540 $SEG/segformer/tests/pcc/test_segformer_for_semantic_segmentation.py $SEG/segformer/tests/pcc/test_segformer_for_image_classification.py $SEG/segformer/tests/pcc/test_segformer_model.py

# SegFormer traced 2CQ e2e (the path the README FPS comes from). Crash/hang check, no PCC bar.
# Weights: same as above, cached.
pt segformer_e2e 600 --timeout=540 "$SEG/segformer/tests/perf/test_e2e_performant.py::test_segformer_e2e"

# MobileNetV2 (models/README: batch 10, N150, 3030 FPS). Not in any CI job; only an
# uncalled shell function (run_python_model_tests.sh) still names it.
# Threshold: PCC 0.944 with pretrained weights, batch 10.
# Weights: download.pytorch.org mobilenet_v2-b0353104.pth via wget, 14 MB.
pt mobilenetv2_pcc 420 --timeout=360 $CLS/mobilenetv2/tests/pcc/test_mobilenetv2.py

# MobileNetV2 traced 2CQ e2e. Threshold: PCC 0.94 on last output.
pt mobilenetv2_e2e 600 --timeout=540 "$CLS/mobilenetv2/tests/perf/test_perf_e2e_mobilenetv2.py::test_mobilenetv2_e2e"

# Sentence-BERT (models/README: batch 8, N150, 433 sent/s). Not in any CI job. Uses
# split_query_key_value_and_split_heads (ProgramDescriptor port 2026-09-25) and fused embedding (port 2026-09-22).
# Threshold: PCC 0.986. Weights: emrecan/bert-base-turkish-cased-mean-nli-stsb-tr, public HF, 0.44 GB.
pt sbert_pcc 600 --timeout=540 $WH/sentence_bert/tests/pcc/test_ttnn_sentencebert_model.py

# Sentence-BERT traced e2e (batch 8, seq 384) and the old single-card demo test.
# Demo asserts its own cosine-similarity bar; e2e is crash/hang only.
pt sbert_e2e 600 --timeout=540 "$WH/sentence_bert/tests/perf/test_sentence_bert_e2e_performant.py::test_e2e_performant_sentencebert"
pt sbert_demo 600 --timeout=540 "$WH/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference"
hfrm emrecan/bert-base-turkish-cased-mean-nli-stsb-tr

# Vanilla UNet (models/README: 480x640, N150, 60 FPS vs target 240). Not in any CI job.
# Threshold: PCC 0.977 (VANILLA_UNET_PCC_WH).
# Weights: github.com/mateuszbuda/brain-segmentation-pytorch unet.pt via wget, 31 MB.
pt vanilla_unet_pcc 600 --timeout=540 $SEG/vanilla_unet/tests/test_unet_model.py

# Vanilla UNet traced e2e, 32 iterations. Threshold: PCC 0.977 plus perf report (190 FPS expectation).
pt vanilla_unet_e2e 900 --timeout=840 "$SEG/vanilla_unet/tests/test_unet_perf.py::test_vanilla_unet_perf_e2e"

# VGG-UNet (models/README: 256x256, N150, 198 FPS). Not in any CI job.
# Threshold: PCC 0.98. Weights: random (pretrained variant is commented out in the test), 0 MB.
pt vgg_unet_pcc 420 --timeout=360 $SEG/vgg_unet/wormhole/tests/pcc/test_vgg_unet.py
pt vgg_unet_e2e 600 --timeout=540 "$SEG/vgg_unet/wormhole/tests/perf/test_e2e_performant.py::test_vgg_unet_e2e"

# UFLD v2 (models/README: 320x800, N150, 365 FPS). Not in any CI job.
# Thresholds: basic block PCC 0.99, full model PCC 0.989. Weights: random, 0 MB.
pt ufld_v2_pcc 600 --timeout=540 $SEG/ufld_v2/wormhole/tests/pcc/test_ttnn_ufld_v2.py
pt ufld_v2_e2e 600 --timeout=540 "$SEG/ufld_v2/wormhole/tests/perf/test_ufld_v2_e2e_performant.py::test_ufldv2_e2e_performant"

# EfficientNet-B0 (README: N150, 157 FPS). Removed from CI by #52318.
# Threshold: PCC 0.92. Weights: efficientnet_pytorch release (github), ~21 MB.
pt effnetb0_pcc 420 --timeout=360 $EXP/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py

# Swin-S (README: N150). Removed from CI by #52318. Runs both random and pretrained variants.
# Thresholds: PCC 0.98 pretrained, 0.99 random (a comment in the test records a past value
# of 0.9754, so the margin is thin). Weights: torchvision swin_s IMAGENET1K_V1, 190 MB.
pt swin_s_pcc 900 --timeout=840 $EXP/swin_s/tests/pcc/test_ttnn_swin_transformer.py

# Swin-V2-S (README: N150). Removed from CI by #52318.
# Threshold: PCC 0.98. Weights: torchvision swin_v2_s IMAGENET1K_V1, 190 MB.
pt swin_v2_pcc 900 --timeout=840 $EXP/swin_v2/tests/pcc/test_ttnn_swin_v2_s.py

# VoVNet (README: N150). Removed from CI by #52318. Class (c): the bar is PCC 0.79, so read
# the logged PCC, not the pass/fail. Weights: timm/ese_vovnet19b_dw.ra_in1k, public HF, 30 MB.
pt vovnet_pcc 420 --timeout=360 $EXP/vovnet/tests/pcc/test_tt_vovnet.py

# Stable Diffusion 1.4 (models/README: 512x512, N150, 4.83 s/image). Removed from CI by #52318.
# Uses create_qkv_heads_from_separate_tensors, split_query_key_value_and_split_heads, concatenate_heads
# (all ProgramDescriptor ports of 2026-09-25). Threshold: UNet 512x512 PCC 0.995.
# Weights: CompVis/stable-diffusion-v1-4, public HF, ~5.5 GB (safetensors unet+vae+clip+safety checker).
need 8 && pt sd14_unet 1500 $NT $SD/tests/test_unet_2d_condition_model.py

# SD 1.4 VAE decoder. Threshold: PCC 0.99. Weights: cached from the line above.
need 3 && pt sd14_vae 900 $NT $SD/tests/vae/test_vae.py

# SD 1.4 full 50-step image vs torch pipeline (torch 50 steps on host CPU first, slow).
# Threshold: image PCC 0.935. The test carries a 600 s marker, disabled here by NT.
need 3 && pt sd14_demo 2400 $NT "$SD/tests/test_demo.py::test_demo_sd"
hfrm CompVis/stable-diffusion-v1-4
el

# =============================================================================
# GROUP B. Shipped models that are in N150 CI, but whose accuracy-bearing tests
# are skipped in CI, or sit next to CI-less siblings.
# =============================================================================

# Whisper (root README featured model on N150). CI runs only conditional generation.
# Audio classification is skipped whenever CI=true ("redundant testing"), so it has no coverage at all.
# No threshold in the test; the predicted language is only logged. Crash/hang check.
# Weights: sanchit-gandhi/whisper-medium-fleurs-lang-id, public HF, 0.62 GB.
pt whisper_audio_cls 900 --timeout=840 "$WSP/demo/demo.py::test_demo_for_audio_classification_inference"
mx whisper_audio_cls "predicted|language|label"
hfrm sanchit-gandhi/whisper-medium-fleurs-lang-id

# Whisper distil-large-v3 dataset run. In CI, but WER/CER are only logged, never asserted
# (class c): a degraded model stays green. Read the WER. Reference for distil-large-v3 on
# librispeech dummy is a few percent. Weights: distil-whisper/distil-large-v3, public HF, 1.5 GB;
# dataset hf-internal-testing/librispeech_asr_dummy, public, small.
pt whisper_wer_distil 1500 --timeout=1440 "$WSP/demo/demo.py::test_demo_for_conditional_generation_dataset" -k "distil"
mx whisper_wer_distil "Error Rate"
hfrm distil-whisper/distil-large-v3

# Whisper large-v3 French to English translation. Skipped whenever CI=true, so no coverage.
# No threshold; BLEU is only logged. Weights: openai/whisper-large-v3, public HF, 3.1 GB;
# dataset google/fleurs (fr_fr streaming + en_us validation), public. Needs the `evaluate` package.
need 6 && pt whisper_translate 1800 --timeout=1740 "$WSP/demo/demo.py::test_demo_for_translation_dataset"
mx whisper_translate "BLEU|Translated output|ModuleNotFound"
hfrm openai/whisper-large-v3

# BGE-large-en-v1.5 (models/demos/wormhole/bge_large_en README: n150, n300). Not in any CI job
# (only BGE-M3 is). #58442 item 2 covers a Blackhole-only import bug, not Wormhole.
# Thresholds: full model PCC 0.94 (class c, loose). Weights: BAAI/bge-large-en-v1.5, public HF, 1.34 GB.
pt bge_large_pcc 900 --timeout=840 $WH/bge_large_en/tests/pcc/test_ttnn_bge_model.py
pt bge_large_e2e 900 --timeout=840 "$WH/bge_large_en/tests/perf/test_bge_e2e_performant.py::test_e2e_performant_bge"
hfrm BAAI/bge-large-en-v1.5

# OWL-ViT (models/demos/wormhole/owl_vit). Not in any CI job. Tests self-skip when HF or the
# sample image cannot be fetched, so "skipped" here means offline, not pass.
# Threshold: PCC 0.95. Weights: google/owlvit-base-patch32, public HF, 0.61 GB.
pt owl_vit 1200 --timeout=1140 $WH/owl_vit/tests/test_ttnn_owl_vit.py
hfrm google/owlvit-base-patch32

# models/demos/bert (README: "Wormhole (n150)"). Its only PCC test is skipif(is_wormhole_b0)
# with reason "Unsupported on WH and BH" and a 0.396 PCC bar, and the demo is in no CI job.
# README claim vs test skip is the suspect. Demo runs 3 variants x 5 iterations, batch 8.
# Weights: phiyodr/bert-large-finetuned-squad2, public HF, 1.34 GB (.bin).
pt bert_ttnn_demo 1200 --timeout=1140 --input-path=models/demos/bert/demo/input_data.json "models/demos/bert/demo/demo.py::test_demo"
mx bert_ttnn_demo "answer|exact|f1"
hfrm phiyodr/bert-large-finetuned-squad2
el

# =============================================================================
# GROUP C. tt_transformers N150 models and modes that CI does not run.
# N150 CI covers: Llama-3.1-8B, Llama-3.2-1B/3B, Mistral-7B-v0.3, Phi-3-mini-128k,
# Gemma-2-2B, Qwen3-0.6B/1.7B. Always performance mode, batch 1 (except 8B and
# Mistral ci-eval-32). Accuracy mode, batch 32 and long context have no N150 job
# for any other model, and none of the "extended list" N150 models is run.
# Already pitched, not repeated: Qwen2.5-0.5B, Qwen2.5-1.5B, Mistral v0.1, Phi-3-4k.
# =============================================================================

# Env vars are passed as bash prefix assignments on the pt call (VAR=x pt ...); bash exports
# them to the pytest child for that call only. `env VAR=x` cannot wrap a shell function.

# TinyLlama-1.1B (README extended list: N150). No CI. Vocab 32000, 22 layers.
# test_model "full" compares every decode step against the HF model on host.
# Threshold: PCC 0.86 in performance mode (class c, loose; read the logged PCC per token).
# Weights: TinyLlama/TinyLlama-1.1B-Chat-v1.0, public HF, 2.2 GB. Test has an 1800 s marker.
need 6 && HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 MESH_DEVICE=N150 pt tinyllama_model_pcc 1500 $NT $TTT/tests/test_model.py -k "full and performance"

# Same model through the user-facing demo, batch 1. No threshold: crash/hang/garbage-text check.
need 6 && HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 MESH_DEVICE=N150 pt tinyllama_demo_b1 900 --timeout=840 $TTT/demo/simple_text_demo.py -k "performance and batch-1"
hfrm TinyLlama/TinyLlama-1.1B-Chat-v1.0

# SmolLM2-1.7B-Instruct (README extended list: N150). No CI. Vocab 49152 (small vocab: if this
# dies in the LM head it is the already-pitched Qwen 0.5B DRAM crash family, do not re-pitch).
# Threshold: PCC 0.86 (test_model full, performance). Weights: public HF, 3.4 GB.
need 8 && HF_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct MESH_DEVICE=N150 pt smollm2_model_pcc 1500 $NT $TTT/tests/test_model.py -k "full and performance"
need 8 && HF_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct MESH_DEVICE=N150 pt smollm2_demo_b1 900 --timeout=840 $TTT/demo/simple_text_demo.py -k "performance and batch-1"
hfrm HuggingFaceTB/SmolLM2-1.7B-Instruct

# Qwen2.5-3B-Instruct (README extended list: N150). No CI; the only Qwen2.5 CI job is 7B on N300.
# Threshold: PCC 0.86 (test_model full, performance). Weights: Qwen/Qwen2.5-3B-Instruct, public HF, 6.2 GB.
need 12 && HF_MODEL=Qwen/Qwen2.5-3B-Instruct MESH_DEVICE=N150 pt qwen25_3b_model_pcc 1800 $NT $TTT/tests/test_model.py -k "full and performance"
need 12 && HF_MODEL=Qwen/Qwen2.5-3B-Instruct MESH_DEVICE=N150 pt qwen25_3b_demo_b1 1200 --timeout=1140 $TTT/demo/simple_text_demo.py -k "performance and batch-1"
hfrm Qwen/Qwen2.5-3B-Instruct
el

# Phi-3-mini-128k-instruct, ACCURACY mode token matching. CI runs only performance mode.
# PERF.md accuracy table claims top-1 94 / top-5 99 on N150; the enforced bar is the
# performance-mode one (model_targets.yaml: top1 89, top5 99, minus 0.5), so a 5 point
# accuracy-mode loss stays green even if someone ran it (class b + c).
# Weights: microsoft/Phi-3-mini-128k-instruct, public HF, 7.6 GB, plus tt cache.
need 16 && HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150 pt phi3_128k_acc_tokmatch 1500 --timeout=1440 $TTT/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
mx phi3_128k_acc_tokmatch "Top1 Accuracy|Top5 Accuracy|centralized"

# Phi-3-mini-128k batch 32 (README table advertises batch 32 for every N150 LLM). No N150 CI
# job runs ci-32 for any model. No accuracy bar; crash/hang/garbage check plus t/s.
need 8 && HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150 pt phi3_128k_b32 1200 --timeout=1140 $TTT/demo/simple_text_demo.py -k "performance and ci-32"
mx phi3_128k_b32 "tok/s|t/s|special tokens"

# Phi-3-mini-128k long context, 16k prompt (max_seq_len 32k). No CI job runs any long-context
# id on any device. N150 prefill chunk for this model is 32k. No accuracy bar.
need 8 && HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150 pt phi3_128k_long16k 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "performance and long-context-16k and not ci-long"
mx phi3_128k_long16k "tok/s|t/s|TTFT|special tokens"
hfrm microsoft/Phi-3-mini-128k-instruct

# Phi-3.5-mini-instruct. PERF.md lists N150 speed rows for "Phi3.5-mini" with blank accuracy
# cells, a reference file exists (tests/reference_outputs/Phi-3.5-mini-instruct.refpt), and the
# model has an N150 prefill-chunk entry, but there is no CI job and no model_targets entry.
# Expect the test to END with ValueError "Could not find centralized accuracy targets" after
# printing Top1/Top5; that error is not the finding, the printed accuracy is. For scale:
# Phi-3-mini-128k is 89/99 in performance mode.
# Weights: microsoft/Phi-3.5-mini-instruct, public HF, 7.6 GB.
need 16 && HF_MODEL=microsoft/Phi-3.5-mini-instruct MESH_DEVICE=N150 pt phi35_perf_tokmatch 1500 --timeout=1440 $TTT/demo/simple_text_demo.py -k "performance and ci-token-matching"
mx phi35_perf_tokmatch "Top1 Accuracy|Top5 Accuracy|centralized"
hfrm microsoft/Phi-3.5-mini-instruct
el

# SegFormer sub-module PCC tests (lower value, cheap; localises a full-model failure above).
# Thresholds: per-module PCC 0.9x in each file. Weights: cached segformer-b0, ~40 MB.
pt segformer_pcc_modules 1200 --timeout=600 $SEG/segformer/tests/pcc --deselect $SEG/segformer/tests/pcc/test_segformer_for_semantic_segmentation.py --deselect $SEG/segformer/tests/pcc/test_segformer_for_image_classification.py --deselect $SEG/segformer/tests/pcc/test_segformer_model.py
el

# =============================================================================
# GROUP D (RUN_BIG=1). Public weights of 11 to 15 GB each. About 60 to 75 min.
# =============================================================================
if [ "$RUN_BIG" = 1 ]; then

# Falcon-7B (models/README: N150, 18.5 t/s/u). N150 CI runs one line only:
# default_mode_1024_stochastic, which has no output check at all. The greedy_verify id compares
# against expected_greedy_output.json and is not in CI. Uses nlp_create_qkv_heads_falcon7b
# (ProgramDescriptor port 2026-09-25). Threshold: exact match of expected greedy output.
# Weights: tiiuae/falcon-7b-instruct, public HF, 14.4 GB safetensors, plus tt cache.
need 30 && pt falcon7b_greedy_verify 2400 $NT --input-method=json --input-path=models/demos/falcon7b_common/demo/input_data.json "models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo" -k "default_mode_1024_greedy_verify"
mx falcon7b_greedy_verify "expected|match|differ"

# Falcon-7B perplexity on device, seq 128, prefill and decode. Not in CI.
# Thresholds: prefill ppl 20.00 top1 0.41 top5 0.66; decode ppl 20.25 top1 0.40 top5 0.66.
need 12 && pt falcon7b_ppl_128 2400 $NT models/demos/falcon7b_common/tests/perplexity/test_perplexity_falcon.py -k "prefill_seq128_dram or decode_128_l1_sharded"
mx falcon7b_ppl_128 "perplexity|ppl|top1|top5"

# Falcon-7B 32-layer end-to-end PCC, prefill 128 and decode batch 32. Not in CI.
# Thresholds: prefill output PCC 0.97 (k 0.99, v 0.96); decode 0.86 (class c, loose).
need 12 && pt falcon7b_e2e_pcc 2400 $NT models/demos/falcon7b_common/tests/test_falcon_end_to_end.py -k "prefill_seq128_bf16_dram or decode_batch32_128_bf16_dram"
hfrm tiiuae/falcon-7b-instruct
el

# Mamba-2.8B (models/README: N150, 14.1 t/s/u). N150 CI runs only the demo (perf numbers, no
# accuracy). Full-model PCC and device perplexity are not in CI.
# Thresholds: PCC 0.9759 (prefill 32), 0.9604 (prefill 128), 0.9647 (decode, 64 layers), 0.9954 (1 layer).
# Weights: state-spaces/mamba-2.8b, public HF, 11.1 GB.
need 26 && pt mamba_model_pcc 2400 $NT "$WH/mamba/tests/test_mamba_model.py::test_inference"

# Mamba device perplexity on wikitext-2. Not in CI (the host-reference twin is skipped in CI).
# Thresholds: decode 64: ppl 28.8 top1 0.366 top5 0.619; decode 128: 20.8 / 0.400 / 0.660;
# prefill 64: 27.0 / 0.364 / 0.623; prefill 128: 20.4 / 0.401 / 0.659.
need 12 && pt mamba_ppl 3600 $NT "$WH/mamba/tests/test_mamba_perplexity.py::test_mamba_perplexity"
mx mamba_ppl "Perplexity:|Top-5|Negative log"
hfrm state-spaces/mamba-2.8b
el

# Mistral-7B-Instruct-v0.3, ACCURACY mode token matching. HF API reports this repo as not gated
# on 2026-10-10; if the download 401s, treat it as gated. CI runs performance mode only, and
# test_model "full" is skipped for Mistral-7B (issue #19806), so full-model accuracy mode has no
# gate. PERF.md accuracy table: top-1 96 / top-5 100; enforced bar: 95 / 99 minus 0.5.
# Weights: mistralai/Mistral-7B-Instruct-v0.3, 14.5 GB (repo is 29 GB if fully snapshotted).
need 40 && HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 MESH_DEVICE=N150 pt mistral7b_acc_tokmatch 2400 --timeout=2340 $TTT/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
mx mistral7b_acc_tokmatch "Top1 Accuracy|Top5 Accuracy|centralized"
hfrm mistralai/Mistral-7B-Instruct-v0.3

# Depth Anything V2 Large (experimental README: N150, PCC > 0.99, 15 FPS). No CI.
# Threshold: PCC 0.99. Weights: depth-anything/Depth-Anything-V2-Large-hf, public HF, 1.34 GB.
need 5 && pt depth_anything_v2_pcc 1200 --timeout=1140 $EXP/depth_anything_v2/tests/test_depth_anything_v2_pcc.py
hfrm depth-anything/Depth-Anything-V2-Large-hf
el
fi

# =============================================================================
# GROUP E (RUN_GATED=1). GATED: needs an HF token with the licence accepted.
# =============================================================================
if [ "$RUN_GATED" = 1 ]; then

# GATED WEIGHTS (meta-llama). Llama-3.2-1B accuracy mode. CI runs performance mode only (via the
# tttv2 demo). PERF.md accuracy table: top-1 87 / top-5 99; enforced bar is 79 / 97 minus 0.5,
# so an 8 point accuracy-mode loss passes (class c). Weights: 2.5 GB.
need 8 && HF_MODEL=meta-llama/Llama-3.2-1B-Instruct MESH_DEVICE=N150 pt llama32_1b_acc_tokmatch 1200 --timeout=1140 $TTT/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
mx llama32_1b_acc_tokmatch "Top1 Accuracy|Top5 Accuracy"

# GATED WEIGHTS. Llama-3.2-1B batch 32 and 16k long context (PERF.md has N150 rows for both;
# no N150 CI job). No accuracy bar.
need 8 && HF_MODEL=meta-llama/Llama-3.2-1B-Instruct MESH_DEVICE=N150 pt llama32_1b_b32 1200 --timeout=1140 $TTT/demo/simple_text_demo.py -k "performance and ci-32"
need 8 && HF_MODEL=meta-llama/Llama-3.2-1B-Instruct MESH_DEVICE=N150 pt llama32_1b_long16k 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "performance and long-context-16k and not ci-long"
hfrm meta-llama/Llama-3.2-1B-Instruct

# GATED WEIGHTS. Llama-3.2-3B accuracy mode. PERF.md: 96 / 100; enforced bar 89 / 98 minus 0.5.
# Trace prefill is disabled for this model on N150 (supported seq lens = []). Weights: 6.4 GB.
need 14 && HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MESH_DEVICE=N150 pt llama32_3b_acc_tokmatch 1500 --timeout=1440 $TTT/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
mx llama32_3b_acc_tokmatch "Top1 Accuracy|Top5 Accuracy"
hfrm meta-llama/Llama-3.2-3B-Instruct

# GATED WEIGHTS. Llama-3.2-11B text path on N150. PERF.md and model_targets.yaml both carry an
# N150 row (top-1 90 / top-5 98) but the only CI job is on T3K. Weights: about 21 GB.
need 45 && HF_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct MESH_DEVICE=N150 pt llama32_11b_perf_tokmatch 2400 --timeout=2340 $TTT/demo/simple_text_demo.py -k "performance and ci-token-matching"
mx llama32_11b_perf_tokmatch "Top1 Accuracy|Top5 Accuracy"
hfrm meta-llama/Llama-3.2-11B-Vision-Instruct

# GATED DATASET (ILSVRC/imagenet-1k). MobileNetV2 ImageNet demo, the retired single-card demo.
# Threshold: top-1 accuracy >= 0.68. Weights: 14 MB, cached from group A.
pt mobilenetv2_imagenet_demo 900 --timeout=840 "$CLS/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo"
fi

el
echo "logs in $RUN"
for f in "$RUN"/*.log; do printf '%-28s %s\n' "$(basename "$f" .log)" "$(grep -E '(passed|failed|error|skipped).* in [0-9.]+s' "$f" | tail -1 | cut -c1-90)"; done

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=0
export VLLM_DISABLE_CASCADE_ATTN=1
LOGGING_LEVEL=NOTSET
EXP="No_offload_at_transition" # <---- Change this to the experiment name 
ROOT="/home/xinyuema/vllm"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
NEW_LOG="${ROOT}/configs/test${EXP}.json"          # will be overwritten per-run
OUT_DIR="${ROOT}/outputs/benchmark/${EXP}"
mkdir -p "${OUT_DIR}"

CFG_DIR="${ROOT}/scripts/benchmark/test_traces"
CFG_LIST=(S00)

for T in "${CFG_LIST[@]}"; do
    RUN_DIR="${OUT_DIR}/${T}"           # one sub-folder per cfg
    mkdir -p "${RUN_DIR}"

    # regenerate logging config for this run
sed  -e '15s#"level": *"INFO"#"level\": \"'"${LOGGING_LEVEL}"'"#' \
     -e '16s#"filename":.*#"filename\": \"'"${OUT_DIR}/${T}"'/vllm_msg.log"#' \
     "${BASE_LOG}" > "${NEW_LOG}"
    echo "Logging config: ${OUT_DIR}/${T}/vllm_msg.log"
    export VLLM_LOGGING_CONFIG_PATH="${NEW_LOG}"
    : > "${RUN_DIR}/vllm_msg.log"

    # Change option below to match with the experiment name

    python ../examples/test_distN.py \
        --config_file="${CFG_DIR}/${T}.json" \
        --prefetch_mode=none \
        --prefetch_distance=0 \
        --flattened_cache=true \
        --merge-prefetch-buffer=true \
        --pause-and-resume \
        --output_log=${OUT_DIR}/${T}/outputs.log
done
# Shared GPU-topology defaults. Sourced, not executed.
#
# Every run script defaults to the paper's 8 GPUs / TP=4. Relying on the caller
# to remember `N_GPUS=1 TP_SIZE=1 GPU_MEM_UTIL=0.35 bash ...` on every single
# invocation is how a re-run ends up asking for 8 GPUs on a 1-GPU box, or
# handing vLLM a 0.6 pool it cannot share with a resident optimizer. Detect
# once, here, so a bare `bash run/<script>.sh` is correct on any node.
# An explicit N_GPUS / TP_SIZE / GPU_MEM_UTIL from the environment still wins.

if [ -z "${N_GPUS:-}" ]; then
    N_GPUS=$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    # empty (python3 missing / import failed / no output) counts as one card,
    # never as "unset" — an empty N_GPUS silently falls back to the 8-GPU
    # default further down and dies in verl's resource-pool check.
    case "${N_GPUS}" in ""|0|*[!0-9]*) N_GPUS=1 ;; esac
fi
if [ -z "${TP_SIZE:-}" ]; then
    TP_SIZE=1
    for _t in 4 2; do
        if [ $((N_GPUS % _t)) -eq 0 ] && [ "${N_GPUS}" -ge "${_t}" ]; then TP_SIZE=${_t}; break; fi
    done
    unset _t
fi

# vLLM reserves gpu_memory_utilization x VRAM as a pool for the whole run, and
# verl only moves the FSDP model off the GPU during generation when
# param_offload=True. On one card the paper's 0.6 therefore has to coexist with
# weights + gradients + AdamW state: measured 67.7 of ~73 addressable GiB for
# 1.5B, which OOMs on any fragmentation. Lower it when there is a single GPU.
# This is an inference-engine memory knob only; it does not change any result.
if [ "${N_GPUS}" -eq 1 ] && [ -z "${GPU_MEM_UTIL:-}" ]; then
    GPU_MEM_UTIL=0.35
    echo "[gpu] single GPU: GPU_MEM_UTIL=${GPU_MEM_UTIL} (the paper's 0.6 does not fit"
    echo "[gpu]   alongside the resident optimizer state; results are unaffected)"
fi
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}

export N_GPUS TP_SIZE GPU_MEM_UTIL

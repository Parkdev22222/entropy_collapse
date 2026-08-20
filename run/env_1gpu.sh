# Shared environment for the single-A100 experiment session.
# Sourced by the smoke test and by every arm so that the machine-specific
# settings live in one place instead of being retyped per run.
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
# 255 visible cores make torch's CPU pools thrash (a 6s test suite took >10min
# at the default thread count). Cap them for every child process.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export N_GPUS=1
export TP_SIZE=1
export STEERF_ROOT=/workspace/entropy_collapse
export STEER_ROOT=${STEERF_ROOT}/third_party/STEER
# verl's tensorboard adapter reads this; without it the writer lands in a
# relative `tensorboard_log/` next to whatever cwd the run was launched from.
# One tree for the whole project so arms sit side by side in the UI.
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${STEERF_ROOT}/tensorboard_log/${PROJECT_NAME:-STEER-F}/${RUN_TAG:-run}}
export MODEL_PATH=${MODEL_PATH:-/workspace/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}

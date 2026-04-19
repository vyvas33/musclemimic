#!/usr/bin/env bash
# One-shot bootstrap for training MuscleMimic on an EC2 GPU instance.
#
# Assumes: Ubuntu 22.04 x86_64 with NVIDIA driver + CUDA already installed
# (e.g. AWS "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)").
#
# Usage (from the repo root on the EC2 instance):
#   HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash scripts/ec2_setup.sh
#
# Both tokens are optional. Skip HF_TOKEN only if you won't use gated datasets
# or hf:// checkpoints. Skip WANDB_API_KEY only if you'll train with
# wandb.mode=disabled.

set -euo pipefail

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m  %s\n' "$*" >&2; exit 1; }

# --- 1. Sanity checks ------------------------------------------------------
[[ "$(uname -s)" == "Linux" ]]   || die "This script must run on Linux (you're on $(uname -s))."
[[ "$(uname -m)" == "x86_64" ]]  || die "MuscleMimic CUDA wheels require x86_64 (you're on $(uname -m))."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — use a GPU AMI with NVIDIA drivers preinstalled."
nvidia-smi -L | head -n1 | sed 's/^/[setup] GPU: /'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -f pyproject.toml ]] || die "pyproject.toml not found in $REPO_ROOT — run this from inside the repo."

# --- 2. System packages ----------------------------------------------------
# libGL/libEGL are needed by mujoco's Python bindings even for headless runs.
if command -v apt-get >/dev/null; then
  log "Installing system libraries (libGL, libEGL, git, curl)…"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git curl ca-certificates libgl1 libegl1 libglfw3 libosmesa6 >/dev/null
fi

# --- 3. uv -----------------------------------------------------------------
if ! command -v uv >/dev/null; then
  log "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# --- 4. Python deps with CUDA JAX ------------------------------------------
log "Running uv sync --extra cuda (this downloads ~several GB; be patient)…"
uv sync --extra cuda

# --- 5. Auth ---------------------------------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
  log "Logging in to Hugging Face…"
  uv run hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  warn "HF_TOKEN not set — skipping HF login. Gated datasets (demo cache, hf:// checkpoints) will fail."
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  log "Logging in to Weights & Biases…"
  uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null
else
  warn "WANDB_API_KEY not set — pass wandb.mode=disabled when launching training, or export the key."
fi

# --- 6. GPU smoke test -----------------------------------------------------
log "Verifying JAX sees the GPU…"
uv run python -c "
import jax
devs = jax.devices()
print('[setup] JAX backend:', jax.default_backend())
print('[setup] JAX devices:', devs)
assert any(d.platform == 'gpu' for d in devs), 'No GPU device found by JAX'
"

# --- Done ------------------------------------------------------------------
cat <<EOF

[setup] Done. Next steps (examples):

  # Pull the demo cache (needs HF_TOKEN + dataset access approval):
  uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"

  # Launch training inside tmux so it survives SSH disconnects:
  tmux new -s train
  uv run fullbody/experiment.py --config-name=conf_fullbody_demo
  # detach: Ctrl-b d    reattach later: tmux attach -t train

  # From your Mac, pull results back:
  rsync -avz --progress ec2-user@<host>:$REPO_ROOT/outputs ./
  rsync -avz --progress ec2-user@<host>:$REPO_ROOT/wandb   ./
EOF

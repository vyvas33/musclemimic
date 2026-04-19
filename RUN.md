# RUN.md — Training MuscleMimic end-to-end

Practical walkthrough for cloning this repo and training a policy. Covers two
execution environments:

- **Option A:** Your own Linux machine with an NVIDIA GPU
- **Option B:** AWS EC2 GPU instance (if you don't have a local NVIDIA GPU)

Evaluation works on **macOS or Linux** regardless of where training ran.

---

## 0. Prerequisites (both options)

- **HuggingFace account** with access approved to
  [amathislab/demo_dataset](https://huggingface.co/datasets/amathislab/demo_dataset)
  (needed for the demo motion cache) — create a token at
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- **Weights & Biases account** (optional, but training configs default to
  `wandb.mode=online`). Get your API key from
  [wandb.ai/authorize](https://wandb.ai/authorize).

Export both as env vars wherever you'll run training:

```bash
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=xxx
```

---

## Option A — Train on your own Linux + NVIDIA GPU machine

Requirements: Linux x86_64, NVIDIA driver installed (`nvidia-smi` works), CUDA 12.

```bash
# 1. Clone
git clone https://github.com/vyvas33/musclemimic.git
cd musclemimic

# 2. One-shot setup: installs uv, syncs CUDA deps, logs into HF + W&B
#    and verifies JAX sees the GPU. Idempotent — safe to re-run.
HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash scripts/setup_linux.sh

# 3. Download the demo motion cache
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"

# 4. Train (single-motion walk config — see the table below for alternatives)
uv run fullbody/experiment.py --config-name=conf_fullbody_walk
```

If you prefer the manual path (no setup script), the equivalent commands are:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cuda
uv run hf auth login --token "$HF_TOKEN" --add-to-git-credential
uv run wandb login "$WANDB_API_KEY"
```

---

## Option B — Train on AWS EC2

### B1. Launch a GPU instance

- **AMI:** *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)* —
  NVIDIA drivers + CUDA are preinstalled, which avoids hours of driver wrangling.
- **Instance type:** `g5.xlarge` (A10G, ~$1/hr) is a good starting point.
  Bump to `g5.2xlarge`+ if you hit OOM on fullbody configs with high `num_envs`.
- **Storage:** bump the root EBS volume to **≥80 GB**. A full training run with
  checkpoints + wandb + validation videos easily consumes 10–20 GB, and the
  venv + JAX cache take another few GB.
- **Security group:** inbound SSH (port 22) from your IP is all you need.
- **Key pair:** download the `.pem` file, `chmod 400 ~/.ssh/your-key.pem`.

Add an entry to `~/.ssh/config` on your local machine for convenience:

```
Host AWS-EC2
    HostName <public-ipv4-or-dns>
    User ubuntu
    IdentityFile ~/.ssh/your-key.pem
```

### B2. Get the repo onto the instance

Either `git clone` on the instance (easiest) or `rsync` from your Mac:

```bash
# Option B2a: clone directly on the instance
ssh AWS-EC2
git clone https://github.com/<your-username>/musclemimic.git
cd musclemimic

# Option B2b: rsync from your local machine (useful if you have local edits)
rsync -avz --progress \
  --exclude .venv --exclude outputs --exclude wandb --exclude checkpoints \
  --exclude .git/objects \
  ./ AWS-EC2:~/musclemimic/
```

### B3. Run the bootstrap and start training

```bash
ssh AWS-EC2
cd ~/musclemimic
HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash scripts/setup_linux.sh

# Demo cache
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"

# Launch training inside tmux so it survives SSH disconnects
tmux new -s train
uv run fullbody/experiment.py --config-name=conf_fullbody_walk
# detach:   Ctrl-b  then  d
# reattach: tmux attach -t train
```

To silence the noisy TPU-probe log line, prefix with `JAX_PLATFORMS=cuda`.

### B4. Pull results back to your local machine

From your Mac (run when training is done, or any time — rsync is incremental):

```bash
rsync -av --progress --partial AWS-EC2:~/musclemimic/outputs/     ./outputs/
rsync -av --progress --partial AWS-EC2:~/musclemimic/checkpoints/ ./checkpoints/
rsync -av --progress --partial AWS-EC2:~/musclemimic/wandb/       ./wandb/
```

Drop `-z` on fast connections; checkpoint files are dense floats that don't
compress well.

### B5. **Stop the instance when you're done**

Billing continues while the instance is running. From the EC2 console or CLI:

```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxx
```

Stopping preserves the EBS volume (and your data). Terminate only if you're
fully finished and have pulled everything back.

---

## Evaluation (macOS or Linux)

Evaluation runs on CPU MuJoCo with a viewer, so your Mac works fine:

```bash
# On macOS, use mjpython for viewer-based commands. On Linux, just python.
uv run mjpython fullbody/eval.py \
  --path outputs/<date>/<time>/checkpoints/<run-id>/checkpoint_<N> \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --stochastic --eval_seed 0 --n_steps 1000 --mujoco_viewer
```

Swap `--mujoco_viewer` for `--viser_viewer` to use Viser (nicer muscle rendering,
browser-based).

---

## Config quick reference

| Config | Model | Motions | Use for |
|---|---|---|---|
| `conf_bimanual_demo` | MyoBimanualArm | 3 demo motions | Fast first-time test (~10× faster than fullbody) |
| `conf_fullbody_walk` | MyoFullBody | 1 walking motion, 4096 envs, 4-layer nets | Focused single-motion locomotion training (primary example above) |
| `conf_fullbody_demo` | MyoFullBody | 3 demo motions (walk, turn, circle) | Multi-motion demo of the full pipeline |
| `conf_fullbody_gmr_resnet` | MyoFullBody | Full GMR-retargeted dataset | Real production training runs |

Override any config value via Hydra CLI:

```bash
# Shorter run to sanity-check the pipeline
uv run fullbody/experiment.py --config-name=conf_fullbody_walk \
  experiment.total_timesteps=10_000_000 \
  experiment.env_params.num_envs=512 \
  wandb.mode=disabled
```

See `fullbody/conf_fullbody.yaml` for the full set of tunable parameters.

---

## Troubleshooting

- **`INFO: Unable to initialize backend 'tpu'`** — harmless. JAX probes every
  backend at startup. Silence with `JAX_PLATFORMS=cuda`.
- **`^[[B` garbage in terminal** — you pressed arrow keys while training was
  foregrounded. Detach into `tmux` and don't press keys in the raw SSH session.
- **W&B dashboard stays empty for 5+ minutes** — normal. The first log point
  lands only after the first PPO iteration finishes (`num_envs × num_steps`
  transitions + a gradient update), which takes time on the first run while
  kernels JIT-compile.
- **JAX can't find GPU** — check `nvidia-smi` works, and confirm you installed
  with `--extra cuda`, not the plain `uv sync`.
- **OOM on GPU** — lower `experiment.env_params.num_envs` (4096 in
  `conf_fullbody_walk`, 2048 in `conf_fullbody_demo`). Halving it halves GPU
  memory use.

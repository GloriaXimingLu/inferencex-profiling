# InferenceX vLLM Profiling on Nebius — Runbook

A step-by-step guide to reproduce the vLLM wall-time profiling on Nebius H200 GPUs.
Following this end-to-end takes ~1–1.5 h of hands-on time for the 1×H200 (small-model)
path, plus GPU runtime. All helper scripts referenced live in `nebius/`.

> **TL;DR**: install Nebius CLI → auth into your **billing-enabled** tenant → provision an
> H200 → `node_setup.sh` → `calib*.sh` → `analyze.py` → **delete the node**.

> **Note:** infrastructure IDs (tenant / project / subnet) are environment-specific.
> This runbook uses placeholders like `<YOUR_PROJECT_ID>`; see **§9 → Discover your IDs**
> for the one-liners that print yours.

---

## 0. What you get

A real measurement of how long it takes vLLM to run InferenceX benchmark "cells"
(`model × precision × ISL/OSL × tensor-parallel × concurrency`) on H200, plus an
extrapolation to the full matrix. See `WALLTIME_REPORT.md` for results.

There are two paths:
| Path | Node | Models you can profile | Use when |
|---|---|---|---|
| **Quick** | 1×H200 | `gpt-oss-120b` only (fits in 1 GPU at TP=1) | smoke test / fast endpoint, always available |
| **Full** | 8×H200 | all (DeepSeek-V4-Pro, Kimi, MiniMax, gpt-oss) | full matrix; **8-GPU nodes are often capacity-constrained** |

---

## 1. Prerequisites

- Membership in your **company's Nebius tenant** — the one with **billing enabled**
  (a personal/no-billing tenant will deny every provisioning call). Accept the email
  invite first if you haven't.
- A local machine with `ssh`, `scp`, `git`, `python3`, and `curl`.
- A Hugging Face token (`HF_TOKEN`) **only** if you profile gated models
  (`gpt-oss-120b` is open and needs none; `DeepSeek-V4-Pro`/`Kimi` may be gated).
- This repo cloned, including `nebius/` (these scripts). Also clone the upstream
  benchmark repo (not vendored here):

```bash
git clone https://github.com/SemiAnalysisAI/InferenceX
```

---

## 2. Install & authenticate the Nebius CLI

```bash
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="$HOME/.nebius/bin:$PATH"            # add to your shell profile
nebius version
```

Authenticate (opens a browser). **Select your billing-enabled company tenant**, not a
personal one:

```bash
nebius profile create inferencex
# In the browser: log in, and when prompted choose your COMPANY tenant (not personal).
# Keep the default federation endpoint (auth.nebius.com).
```

> ⚠️ **Critical gotcha:** a tenant with **no billing** denies every provisioning call
> with `PermissionDenied` (even creating a tiny instance). You must be on the
> billing-enabled tenant. `nebius iam whoami` lists the tenants you belong to — make
> sure the right one is there.

Point the CLI at your project (see §9 to find the ID):

```bash
nebius config set parent-id <YOUR_PROJECT_ID>
```

---

## 3. Create an SSH key and prepare cloud-init

```bash
ssh-keygen -t ed25519 -f ~/.ssh/nebius_inferencex -N "" -C "inferencex-$(whoami)"
```

Edit `nebius/cloud-init.yaml` and replace the placeholder `ssh_authorized_keys` value
with **your** public key:

```bash
cat ~/.ssh/nebius_inferencex.pub   # paste this into cloud-init.yaml
```

---

## 4. Provision a node

Set your environment-specific variables (see §9 to discover the IDs):

```bash
PROJECT=<YOUR_PROJECT_ID>        # nebius iam project list --parent-id <YOUR_TENANT_ID>
SUBNET=<YOUR_SUBNET_ID>          # nebius vpc subnet list --parent-id <YOUR_PROJECT_ID>
IMAGES=<REGION_PUBLIC_IMAGES>    # Nebius public image catalog for your region (see §9)
CI=$(cat nebius/cloud-init.yaml)
```

### Quick path — 1×H200 (always available)

```bash
nebius compute instance create \
  --parent-id $PROJECT --name inferencex-h200-1g \
  --resources-platform gpu-h200-sxm --resources-preset 1gpu-16vcpu-200gb \
  --boot-disk-attach-mode READ_WRITE --boot-disk-managed-disk-name ix-1g-boot \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 \
  --boot-disk-managed-disk-source-image-family-parent-id $IMAGES \
  --boot-disk-managed-disk-size-gibibytes 500 \
  --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET\",\"ip_address\":{},\"public_ip_address\":{}}]" \
  --cloud-init-user-data "$CI"
```

### Full path — 8×H200 (for the big models)

Same command with `--resources-preset 8gpu-128vcpu-1600gb` and
`--boot-disk-managed-disk-size-gibibytes 2000` (DeepSeek-V4-Pro weights are ~670 GB).

> ⚠️ **8-GPU nodes frequently fail with `NotEnoughResources` (schedule timeout)** even
> though quota shows free — that's *live capacity*, not quota. If one region fails, try
> another; a failed create costs **$0** (nothing is provisioned) but leaves a STOPPED
> instance — delete it before retrying (see §8).

Get the public IP:

```bash
nebius compute instance list --parent-id $PROJECT --format json \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['items'][0]['status']['network_interfaces'][0]['public_ip_address']['address'].split('/')[0])"
```

---

## 5. Set up the node

```bash
IP=<public-ip-from-above>
OPTS=(-i ~/.ssh/nebius_inferencex -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes)
# Wait ~30-60s for cloud-init, then:
scp "${OPTS[@]}" nebius/node_setup.sh nebius/calib.sh nebius/calib_1g.sh ubuntu@$IP:/tmp/
ssh "${OPTS[@]}" ubuntu@$IP 'bash /tmp/node_setup.sh'
```

`node_setup.sh` installs Docker + the NVIDIA container toolkit, runs a GPU smoke test,
and clones InferenceX to `/opt/inferencex`.

> ⚠️ **zsh gotcha (macOS default shell):** do **not** pack the whole ssh command into a
> string variable (`SSH="ssh -i ..."; $SSH ...`) — zsh won't word-split it. Use an array
> `OPTS=(...)` and `"${OPTS[@]}"` as shown. Also never name a variable `status` in zsh
> (it's read-only).

---

## 6. Run the profiling

Profiling runs **detached** under `nohup` so an SSH drop can't kill it.

### 1×H200 (gpt-oss-120b):

```bash
ssh "${OPTS[@]}" ubuntu@$IP 'nohup bash /tmp/calib_1g.sh > /opt/results/calib.out 2>&1 & echo started'
# watch progress:
ssh "${OPTS[@]}" ubuntu@$IP 'tail -f /opt/results/calib.out'      # Ctrl-C to stop tailing
```

### 8×H200 (gpt-oss + DeepSeek-V4-Pro):

```bash
# export HF_TOKEN first if the big models are gated:
ssh "${OPTS[@]}" ubuntu@$IP "HF_TOKEN=hf_xxx nohup bash /tmp/calib.sh > /opt/results/calib.out 2>&1 & echo started"
```

**What the scripts measure** (per cell, version-independent — they poll vLLM's
`/health`, not log strings):
- `download_s` — one-time model download (cached on disk after first cell)
- `startup_s` — container launch → server ready (weight load + CUDA-graph capture)
- `bench_s` — server ready → cell complete (the InferenceX serving benchmark)

Edit the cell list at the bottom of `calib.sh` / `calib_1g.sh` to change models,
TP, concurrency, or ISL/OSL. Results land in `/opt/results/timings.tsv` plus a
per-cell log.

When you see `===== CALIBRATION COMPLETE =====`, pull the results:

```bash
scp "${OPTS[@]}" -r ubuntu@$IP:/opt/results nebius/
```

---

## 7. Analyze

```bash
python3 -m venv .venv && .venv/bin/pip install pyyaml      # if needed
.venv/bin/python nebius/analyze.py
```

This prints the measured cells and extrapolates to the full H200 vLLM matrix.
Tune the `MODELS` multipliers in `analyze.py` as you gather more real data.

---

## 8. ⚠️ TEAR DOWN (do not skip — GPUs bill by the second)

```bash
IID=$(nebius compute instance list --parent-id $PROJECT --format json \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['items'][0]['metadata']['id'])")
nebius compute instance delete --id $IID

# verify NOTHING is left billing (instances AND disks):
nebius compute instance list --parent-id $PROJECT | grep -c computeinstance
nebius compute disk list      --parent-id $PROJECT | grep -c computedisk
```

Deleting the instance releases its managed boot disk and public IP. Do a final sweep
across all regions you touched.

---

## 9. Reference

### Discover your IDs
Everything below is environment-specific — print yours with:

```bash
nebius iam whoami                                 # your tenant_id (and which tenants you're in)
nebius iam project list --parent-id <TENANT_ID>   # one project per region
nebius vpc subnet list  --parent-id <PROJECT_ID>  # the subnet in that project
nebius compute platform list                      # confirm gpu-h200-sxm + presets
```

The **public image catalog** is a Nebius-shared project, named by region prefix:
`project-e00public-images` (eu-north1), `project-e01public-images` (eu-west1),
`project-u00public-images` (us-central1) — use the one matching your project's region as
`<REGION_PUBLIC_IMAGES>`.

| Field | Value |
|---|---|
| Platform | `gpu-h200-sxm` |
| Presets | `1gpu-16vcpu-200gb`, `8gpu-128vcpu-1600gb` |
| Image family | `ubuntu24.04-cuda13.0` (has the GPU driver; default login user `ubuntu`) |

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PermissionDenied` on any create | On a tenant with no billing | Switch to your billing-enabled tenant (§2) |
| `NotEnoughResources` / schedule timeout | No live 8×H200 capacity | Try another region; delete the STOPPED instance left behind |
| SSH hangs / `Permission denied (publickey)` | cloud-init key not applied / wrong key | Confirm your pubkey is in `cloud-init.yaml`; wait 60s for boot |
| `$SSH ...` → "no such file or directory" | zsh doesn't word-split string vars | Use `OPTS=(...)` array + `"${OPTS[@]}"` |
| `read-only variable: status` | `status` is reserved in zsh | Rename the variable |
| Server never becomes ready | OOM / model too big for the GPU(s) | Check the per-cell log; reduce TP scope or use the 8-GPU node |

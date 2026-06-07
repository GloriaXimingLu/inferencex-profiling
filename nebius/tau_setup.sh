#!/usr/bin/env bash
# Runs ON a tau-bench node. Installs docker + nvidia-container-toolkit (for the
# vLLM server) and host-side python deps (for client.py). The tau-bench-replay
# files (client.py, scheduler.py, synth.py, schedule/) are scp'd to /opt/tau.
set -uo pipefail
echo "===== HOST GPU ====="; nvidia-smi -L || echo "WARN no nvidia-smi"
export DEBIAN_FRONTEND=noninteractive
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -y -qq && sudo apt-get install -y -qq docker.io
fi
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo usermod -aG docker ubuntu 2>/dev/null || true
# host python deps for client.py
sudo apt-get install -y -qq python3-pip 2>/dev/null || true
pip install --break-system-packages -q numpy aiohttp 2>/dev/null || pip3 install --break-system-packages -q numpy aiohttp 2>/dev/null || true
sudo mkdir -p /opt/tau /opt/hfcache /opt/results && sudo chown -R ubuntu:ubuntu /opt/tau /opt/hfcache /opt/results
echo "===== docker GPU test ====="; sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi -L || echo "WARN docker GPU test failed"
echo "===== python deps ====="; python3 -c "import numpy,aiohttp;print('numpy',numpy.__version__,'aiohttp',aiohttp.__version__)" || echo "WARN python deps missing"
echo "===== TAU SETUP DONE ====="

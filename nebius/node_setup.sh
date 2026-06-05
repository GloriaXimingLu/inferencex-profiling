#!/usr/bin/env bash
# Runs ON the H200 node. Installs docker + nvidia-container-toolkit, clones
# InferenceX, preps HF cache + results dirs on the big root disk.
set -uo pipefail
echo "===== HOST GPU ====="; nvidia-smi -L || echo "WARN: no host nvidia-smi"
echo "===== DISK ====="; df -h / | tail -1

export DEBIAN_FRONTEND=noninteractive
if ! command -v docker >/dev/null 2>&1; then
  echo "===== installing docker ====="
  sudo apt-get update -y -qq && sudo apt-get install -y -qq docker.io
fi
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "===== installing nvidia-container-toolkit ====="
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
sudo usermod -aG docker ubuntu 2>/dev/null || true

echo "===== docker GPU smoke test ====="
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi -L \
  || echo "WARN: docker GPU test failed"

echo "===== dirs + repo ====="
sudo mkdir -p /opt/inferencex /opt/hfcache /opt/results
sudo chown -R ubuntu:ubuntu /opt/inferencex /opt/hfcache /opt/results
[ -d /opt/inferencex/.git ] || git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX /opt/inferencex
echo "===== SETUP DONE ====="

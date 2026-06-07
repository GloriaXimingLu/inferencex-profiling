#!/usr/bin/env bash
# Orchestrate ONE tau-bench domain node end-to-end:
#   create 1xH200 (region fallback) -> wait RUNNING+IP -> wait SSH -> scp harness
#   + tau-bench-replay files -> tau_setup -> launch tau_run detached.
# Usage: bash provision_tau_node.sh <domain> <conc> <ntraces> [region]
set -uo pipefail
DOMAIN="$1"; CONC="$2"; NTRACES="$3"; REGION="${4:-west}"
PRESET="${PRESET:-1gpu-16vcpu-200gb}"; DISK="${DISK:-500}"; PREEMPT="${PREEMPT:-0}"; TP="${TP:-1}"
PRE=(); [ "$PREEMPT" = 1 ] && PRE=(--preemptible-on-preemption stop --recovery-policy fail)  # preemptible requires recovery-policy=fail
export PATH="$HOME/.nebius/bin:$PATH" NO_COLOR=1 TERM=dumb
strip(){ sed -E 's/\x1b\[[0-9;]*m//g'; }
ROOT=/Users/ximinglu/Projects/profiling
KEY=$HOME/.ssh/nebius_inferencex
OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes)
CI=$(cat "$ROOT/nebius/cloud-init.local.yaml")
NAME="ix-tau-${DOMAIN//_/-}"
PY=/usr/bin/python3

# Fill in your project/subnet IDs per region (see PROFILING_RUNBOOK.md §9 "Discover your IDs":
#   nebius iam project list --parent-id <TENANT>;  nebius vpc subnet list --parent-id <PROJECT>
# IMG is the Nebius public image catalog for that region (project-<prefix>public-images).
region_cfg(){ case "$1" in
  west)    PID=<YOUR_PROJECT_ID_WEST>;    SUB=<YOUR_SUBNET_ID_WEST>;    IMG=<REGION_PUBLIC_IMAGES_WEST>;;
  central) PID=<YOUR_PROJECT_ID_CENTRAL>; SUB=<YOUR_SUBNET_ID_CENTRAL>; IMG=<REGION_PUBLIC_IMAGES_CENTRAL>;;
  north)   PID=<YOUR_PROJECT_ID_NORTH>;   SUB=<YOUR_SUBNET_ID_NORTH>;   IMG=<REGION_PUBLIC_IMAGES_NORTH>;;
esac; }

CREATED_PID=""
for R in "$REGION" west central; do
  region_cfg "$R"
  echo "[$DOMAIN] create in $R"
  out=$(nebius compute instance create --parent-id "$PID" --name "$NAME" \
    --resources-platform gpu-h200-sxm --resources-preset "$PRESET" "${PRE[@]}" \
    --boot-disk-attach-mode READ_WRITE --boot-disk-managed-disk-name "${NAME}-boot" \
    --boot-disk-managed-disk-type network_ssd \
    --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 \
    --boot-disk-managed-disk-source-image-family-parent-id "$IMG" \
    --boot-disk-managed-disk-size-gibibytes "$DISK" \
    --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUB\",\"ip_address\":{},\"public_ip_address\":{}}]" \
    --cloud-init-user-data "$CI" 2>&1 | strip)
  if echo "$out" | grep -qiE 'not enough|NotEnoughResources'; then
    echo "[$DOMAIN] $R no capacity"
    iid=$(nebius compute instance list --parent-id "$PID" --format json 2>/dev/null | $PY -c "import sys,json;[print(i['metadata']['id']) for i in json.load(sys.stdin).get('items',[]) if i['metadata']['name']=='$NAME']")
    [ -n "$iid" ] && nebius compute instance delete --id "$iid" >/dev/null 2>&1
    continue
  fi
  CREATED_PID="$PID"; CREATED_REGION="$R"; break
done
[ -z "$CREATED_PID" ] && { echo "[$DOMAIN] FAILED: no capacity any region"; exit 1; }

IP=""
for i in $(seq 1 40); do
  IP=$(nebius compute instance list --parent-id "$CREATED_PID" --format json 2>/dev/null | $PY -c "import sys,json
for it in json.load(sys.stdin).get('items',[]):
  if it['metadata']['name']=='$NAME' and it.get('status',{}).get('state')=='RUNNING':
    try: print(it['status']['network_interfaces'][0]['public_ip_address']['address'].split('/')[0])
    except: pass")
  [ -n "$IP" ] && break; sleep 10
done
[ -z "$IP" ] && { echo "[$DOMAIN] NO IP"; exit 1; }
echo "[$DOMAIN] IP=$IP region=$CREATED_REGION pid=$CREATED_PID"

for i in $(seq 1 30); do ssh "${OPTS[@]}" ubuntu@"$IP" 'echo ok' 2>/dev/null | grep -q ok && break; sleep 8; done
ssh "${OPTS[@]}" ubuntu@"$IP" 'sudo mkdir -p /opt/tau && sudo chown ubuntu:ubuntu /opt/tau' 2>/dev/null
scp "${OPTS[@]}" "$ROOT"/nebius/tau_setup.sh "$ROOT"/nebius/tau_lib.sh "$ROOT"/nebius/tau_run.sh ubuntu@"$IP":/tmp/ 2>&1 | grep -v "Permanently added"
scp "${OPTS[@]}" "$ROOT"/tau-bench-replay/client.py "$ROOT"/tau-bench-replay/scheduler.py "$ROOT"/tau-bench-replay/synth.py "$ROOT"/tau-bench-replay/requirements.txt ubuntu@"$IP":/opt/tau/ 2>&1 | grep -v "Permanently added"
scp "${OPTS[@]}" -r "$ROOT"/tau-bench-replay/schedule ubuntu@"$IP":/opt/tau/ 2>&1 | grep -v "Permanently added"
ssh "${OPTS[@]}" ubuntu@"$IP" 'bash /tmp/tau_setup.sh 2>&1 | tail -4'
ssh "${OPTS[@]}" ubuntu@"$IP" "cd /tmp && DOMAIN=$DOMAIN CONC=$CONC NTRACES=$NTRACES TP=$TP MAXSEQS=256 nohup bash tau_run.sh > /opt/results/tau_${DOMAIN}.out 2>&1 & echo launched PID \$!"
echo "[$DOMAIN] LAUNCHED  IP=$IP  PID=$CREATED_PID  REGION=$CREATED_REGION"

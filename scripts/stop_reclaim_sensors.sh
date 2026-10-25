#!/usr/bin/env bash
# Terminate every reclaim-sensor instance (tagged argus-sensor=true) immediately.
#   AWS_PROFILE=argus-demo ./scripts/stop_reclaim_sensors.sh
set -euo pipefail
REGION="${AWS_REGION:-eu-north-1}"
IDS="$(aws ec2 describe-instances --region "$REGION" \
  --filters Name=tag:argus-sensor,Values=true Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [ -z "$IDS" ]; then echo "no sensor instances running"; exit 0; fi
# shellcheck disable=SC2086
aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text

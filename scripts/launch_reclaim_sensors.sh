#!/usr/bin/env bash
# Launch a small, strictly time-capped fleet of tiny Spot instances to collect REAL reclaim labels.
#
# Why: the risk model is trained on a price-spike proxy, and the only two real reclaims on record
# (docs/trace_benchmark_result.md) showed no price signal. CloudTrail logs a BidEvictedEvent for every
# reclaim of an instance in this account; this fleet generates those events on purpose, across
# instance types and AZs, so labels can be pulled with ml/data/pull_reclaim_labels.py.
#
# Cost control (all enforced, not advisory):
#   * every instance runs `shutdown -h +<minutes>` from user-data and is launched with
#     InstanceInitiatedShutdownBehavior=terminate, so it deletes itself after HOURS hours
#   * no public IPv4 address (no hourly address charge), no inbound rules, IMDSv2 only
#   * hard cap on the instance count (MAX_INSTANCES) and a one-line stop script
#   * all instances are tagged argus-sensor=true
# Rough cost: ~$0.01 per instance-hour for these types, so 15 instances x 8 h is about $1 to $2.
#
#   AWS_PROFILE=argus-demo ./scripts/launch_reclaim_sensors.sh
#   AWS_PROFILE=argus-demo ./scripts/stop_reclaim_sensors.sh        # terminate everything early
#
set -euo pipefail

REGION="${AWS_REGION:-eu-north-1}"
HOURS="${HOURS:-8}"
TYPES="${TYPES:-c5.large m5.large t3.medium c6i.large m6i.large}"
AZS="${AZS:-${REGION}a ${REGION}b ${REGION}c}"
MAX_INSTANCES="${MAX_INSTANCES:-15}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ml/data/real"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/sensor_launches_$(date -u +%Y%m%dT%H%M%SZ).csv"

AMI="$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters 'Name=name,Values=al2023-ami-2023*-x86_64' 'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"
SG="$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=default \
  --query 'SecurityGroups[?VpcId!=`null`]|[0].GroupId' --output text)"
echo "AMI $AMI, security group $SG, ${HOURS}h cap" >&2

USERDATA="$(printf '#!/bin/bash\nshutdown -h +%d\n' "$((HOURS * 60))")"
echo "instance_id,instance_type,availability_zone,launched_at_utc,status" > "$LOG"

count=0
for az in $AZS; do
  SUBNET="$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=availability-zone,Values=$az" Name=default-for-az,Values=true \
    --query 'Subnets[0].SubnetId' --output text)"
  for type in $TYPES; do
    if [ "$count" -ge "$MAX_INSTANCES" ]; then break 2; fi
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if out="$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$type" --count 1 \
        --network-interfaces "DeviceIndex=0,SubnetId=$SUBNET,AssociatePublicIpAddress=false,Groups=$SG" \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
        --user-data "$USERDATA" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=argus-sensor,Value=true},{Key=Name,Value=argus-sensor}]' \
        --query 'Instances[0].InstanceId' --output text 2>&1)"; then
      echo "$out,$type,$az,$now,launched" >> "$LOG"
      count=$((count + 1))
      echo "launched $type in $az: $out" >&2
    else
      # A capacity failure is itself a label: that pool had no Spot capacity at this moment.
      reason="$(echo "$out" | grep -o -E 'InsufficientInstanceCapacity|MaxSpotInstanceCountExceeded|SpotMaxPriceTooLow|VcpuLimitExceeded|Unsupported' | head -1)"
      echo ",$type,$az,$now,failed:${reason:-other}" >> "$LOG"
      echo "failed $type in $az: ${reason:-other}" >&2
    fi
  done
done
echo "launched $count instance(s); log: $LOG" >&2

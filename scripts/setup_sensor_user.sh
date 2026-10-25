#!/usr/bin/env bash
# Creates a least-privilege IAM user for the reclaim-sensor campaigns, so the root key is never needed again.
# Run it in AWS CloudShell (console, the >_ icon in the top bar), which is already signed in:
#   bash setup_sensor_user.sh
# It prints an access key ONCE. On your laptop run:  aws configure --profile argus-sensor
# and paste the two values (region eu-north-1, output json). Nobody else needs to see them.
#
# The user can only: launch/terminate Spot instances tagged argus-sensor, describe EC2, read Spot prices and
# placement scores, read CloudTrail events, and read Spot quotas, in the four campaign regions only.
# Remove it afterwards:  bash setup_sensor_user.sh --delete
set -euo pipefail
NAME=argus-sensor
REGIONS="eu-north-1 us-east-1 us-east-2 ap-southeast-1 us-west-2 eu-west-1 eu-central-1 ap-south-1"

if [ "${1:-}" = "--delete" ]; then
  for k in $(aws iam list-access-keys --user-name $NAME --query 'AccessKeyMetadata[].AccessKeyId' --output text); do
    aws iam delete-access-key --user-name $NAME --access-key-id "$k"
  done
  aws iam delete-user-policy --user-name $NAME --policy-name $NAME || true
  aws iam delete-user --user-name $NAME
  echo "deleted $NAME"
  exit 0
fi

cat > /tmp/argus-sensor-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "Describe", "Effect": "Allow",
     "Action": ["ec2:Describe*", "ec2:GetSpotPlacementScores", "cloudtrail:LookupEvents",
                "servicequotas:GetServiceQuota", "servicequotas:ListRequestedServiceQuotaChangeHistory"],
     "Resource": "*",
     "Condition": {"StringEquals": {"aws:RequestedRegion": ["eu-north-1", "us-east-1", "us-east-2", "ap-southeast-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1"]}}},
    {"Sid": "Launch", "Effect": "Allow", "Action": "ec2:RunInstances", "Resource": "*",
     "Condition": {"StringEquals": {"aws:RequestedRegion": ["eu-north-1", "us-east-1", "us-east-2", "ap-southeast-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1"]}}},
    {"Sid": "TagOnLaunch", "Effect": "Allow", "Action": "ec2:CreateTags", "Resource": "*",
     "Condition": {"StringEquals": {"ec2:CreateAction": "RunInstances"}}},
    {"Sid": "TerminateSensorsOnly", "Effect": "Allow", "Action": "ec2:TerminateInstances", "Resource": "*",
     "Condition": {"StringEquals": {"ec2:ResourceTag/argus-sensor": "true"}}}
  ]
}
EOF

aws iam get-user --user-name $NAME >/dev/null 2>&1 || aws iam create-user --user-name $NAME >/dev/null
aws iam put-user-policy --user-name $NAME --policy-name $NAME --policy-document file:///tmp/argus-sensor-policy.json
echo "user $NAME ready with the sensor-only policy"

# The Spot quota caps how many sensors run at once (eu-north-1 hit it at about 14). Ask for 128 vCPUs in each
# region; small increases are often approved automatically within minutes, larger ones can take a day.
for r in $REGIONS; do
  cur=$(aws service-quotas get-service-quota --region "$r" --service-code ec2 --quota-code L-34B43A08 \
        --query 'Quota.Value' --output text 2>/dev/null || echo "?")
  echo "$r Spot vCPU quota now: $cur"
  if [ "$cur" != "?" ] && [ "${cur%.*}" -lt 128 ]; then
    aws service-quotas request-service-quota-increase --region "$r" --service-code ec2 --quota-code L-34B43A08 \
        --desired-value 128 --query 'RequestedQuota.Status' --output text 2>&1 | sed "s/^/  increase request: /" || true
  fi
done

echo
echo "Access key for $NAME (shown once; paste into 'aws configure --profile argus-sensor' on your laptop):"
aws iam create-access-key --user-name $NAME --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text

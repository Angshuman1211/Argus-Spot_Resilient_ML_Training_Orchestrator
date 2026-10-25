#!/usr/bin/env bash
# One-time setup so the sensor campaigns can run on a small EC2 machine instead of a laptop.
# Run it in AWS CloudShell after setup_sensor_user.sh:   bash setup_cloud_runner.sh
# Creates:
#   role argus-campaign-runner (+ instance profile): the sensor permissions, plus read/write on
#       s3://argus-checkpoints-<account>/campaign/ where code, state and results live
#   extra permissions for user argus-sensor: pass that role to the runner machine, read/write the same S3 prefix,
#       and terminate machines tagged argus-runner=true
# Remove it afterwards:   bash setup_cloud_runner.sh --delete
set -euo pipefail
ROLE=argus-campaign-runner
ACCT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=argus-checkpoints-$ACCT
REGIONS='["eu-north-1", "us-east-1", "us-east-2", "ap-southeast-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1"]'

if [ "${1:-}" = "--delete" ]; then
  aws iam remove-role-from-instance-profile --instance-profile-name $ROLE --role-name $ROLE || true
  aws iam delete-instance-profile --instance-profile-name $ROLE || true
  aws iam delete-role-policy --role-name $ROLE --policy-name $ROLE || true
  aws iam delete-role --role-name $ROLE || true
  aws iam delete-user-policy --user-name argus-sensor --policy-name argus-cloud-runner || true
  echo "deleted"
  exit 0
fi

cat > /tmp/trust.json <<'EOF'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
EOF
cat > /tmp/runner.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["ec2:Describe*", "ec2:GetSpotPlacementScores", "ec2:RunInstances"], "Resource": "*",
     "Condition": {"StringEquals": {"aws:RequestedRegion": $REGIONS}}},
    {"Effect": "Allow", "Action": "ec2:CreateTags", "Resource": "*", "Condition": {"StringEquals": {"ec2:CreateAction": "RunInstances"}}},
    {"Effect": "Allow", "Action": "ec2:TerminateInstances", "Resource": "*",
     "Condition": {"StringEquals": {"ec2:ResourceTag/argus-sensor": "true"}}},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": "arn:aws:s3:::$BUCKET/campaign/*"},
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::$BUCKET", "Condition": {"StringLike": {"s3:prefix": "campaign/*"}}},
    {"Effect": "Allow", "Action": "budgets:ViewBudget", "Resource": "arn:aws:budgets::$ACCT:budget/argus-campaign-spend"}
  ]
}
EOF
aws iam get-role --role-name $ROLE >/dev/null 2>&1 || aws iam create-role --role-name $ROLE --assume-role-policy-document file:///tmp/trust.json >/dev/null
aws iam put-role-policy --role-name $ROLE --policy-name $ROLE --policy-document file:///tmp/runner.json
aws iam get-instance-profile --instance-profile-name $ROLE >/dev/null 2>&1 || {
  aws iam create-instance-profile --instance-profile-name $ROLE >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
}

cat > /tmp/user.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "iam:PassRole", "Resource": "arn:aws:iam::$ACCT:role/$ROLE"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": "arn:aws:s3:::$BUCKET/campaign/*"},
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::$BUCKET", "Condition": {"StringLike": {"s3:prefix": "campaign/*"}}},
    {"Effect": "Allow", "Action": "ec2:TerminateInstances", "Resource": "*",
     "Condition": {"StringEquals": {"ec2:ResourceTag/argus-runner": "true"}}}
  ]
}
EOF
aws iam put-user-policy --user-name argus-sensor --policy-name argus-cloud-runner --policy-document file:///tmp/user.json
echo "ready: role $ROLE, bucket prefix s3://$BUCKET/campaign/"

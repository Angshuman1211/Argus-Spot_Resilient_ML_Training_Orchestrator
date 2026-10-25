#!/usr/bin/env bash
# Records AWS's own early-warning signals for our Spot instances, so they can be scored against real reclaims:
#   "EC2 Instance Rebalance Recommendation"  (the free early warning a reviewer asked us to compare against)
#   "EC2 Spot Instance Interruption Warning" (the 2-minute notice, to check its timing against CloudTrail)
# Per campaign region: an EventBridge rule sends these events to the CloudWatch Logs group
# /aws/events/argus-spot-signals (kept 30 days). Events are recorded from the moment this runs; there is no backfill.
# Also lets user argus-sensor read that log group (ml/data/pull_spot_signals.py does the reading).
# Run it in AWS CloudShell:   bash setup_spot_signals.sh        Remove:   bash setup_spot_signals.sh --delete
set -euo pipefail
REGIONS="eu-north-1 us-east-1 us-east-2 ap-southeast-1 us-west-2 eu-west-1 eu-central-1 ap-south-1"
RULE=argus-spot-signals
GROUP=/aws/events/argus-spot-signals
ACCT=$(aws sts get-caller-identity --query Account --output text)

if [ "${1:-}" = "--delete" ]; then
  for r in $REGIONS; do
    aws events remove-targets --region $r --rule $RULE --ids logs >/dev/null 2>&1 || true
    aws events delete-rule --region $r --name $RULE || true
    aws logs delete-log-group --region $r --log-group-name $GROUP || true
    aws logs delete-resource-policy --region $r --policy-name argus-events-to-logs || true
  done
  aws iam delete-user-policy --user-name argus-sensor --policy-name argus-spot-signals || true
  echo "deleted"
  exit 0
fi

PATTERN='{"source": ["aws.ec2"], "detail-type": ["EC2 Instance Rebalance Recommendation", "EC2 Spot Instance Interruption Warning"]}'
for r in $REGIONS; do
  aws logs create-log-group --region $r --log-group-name $GROUP 2>/dev/null || true
  aws logs put-retention-policy --region $r --log-group-name $GROUP --retention-in-days 30
  ARN="arn:aws:logs:$r:$ACCT:log-group:$GROUP"
  # EventBridge needs permission to write into the log group
  aws logs put-resource-policy --region $r --policy-name argus-events-to-logs --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{\"Effect\": \"Allow\", \"Principal\": {\"Service\": [\"events.amazonaws.com\", \"delivery.logs.amazonaws.com\"]},
                    \"Action\": [\"logs:CreateLogStream\", \"logs:PutLogEvents\"], \"Resource\": \"$ARN:*\"}]}" >/dev/null
  aws events put-rule --region $r --name $RULE --event-pattern "$PATTERN" --state ENABLED >/dev/null
  aws events put-targets --region $r --rule $RULE --targets "Id=logs,Arn=$ARN" >/dev/null
  echo "$r: rule $RULE -> $GROUP"
done

cat > /tmp/signals.json <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["logs:FilterLogEvents", "logs:GetLogEvents", "logs:DescribeLogStreams"],
   "Resource": ["arn:aws:logs:*:$ACCT:log-group:$GROUP", "arn:aws:logs:*:$ACCT:log-group:$GROUP:*"]},
  {"Effect": "Allow", "Action": "logs:DescribeLogGroups", "Resource": "arn:aws:logs:*:$ACCT:log-group:*"}]}
EOF
aws iam put-user-policy --user-name argus-sensor --policy-name argus-spot-signals --policy-document file:///tmp/signals.json
echo "ready: argus-sensor can read $GROUP"

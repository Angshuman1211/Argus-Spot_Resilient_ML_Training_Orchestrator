#!/usr/bin/env bash
# Deletes ONLY the argus-eks cluster (its node groups, any load balancers Kubernetes created for it, then the
# cluster). Leaves the VPC, S3 buckets, ECR images, Lambda and IAM roles alone: they cost nothing or are still used
# (the sensor campaign writes to the checkpoints bucket). Run it in AWS CloudShell:   bash delete_eks.sh
# Takes about 15 minutes, mostly waiting for AWS. Recreate later with terraform apply.
set -euo pipefail
R=eu-north-1
C=argus-eks

# load balancers created by Kubernetes Services cost money and block VPC cleanup: remove them first
for arn in $(aws elbv2 describe-load-balancers --region $R --query 'LoadBalancers[].LoadBalancerArn' --output text); do
  if aws elbv2 describe-tags --region $R --resource-arns "$arn" --query 'TagDescriptions[0].Tags[].Key' --output text \
       | tr '\t' '\n' | grep -qx "kubernetes.io/cluster/$C"; then
    echo "deleting load balancer $arn"; aws elbv2 delete-load-balancer --region $R --load-balancer-arn "$arn"
  fi
done
for lb in $(aws elb describe-load-balancers --region $R --query 'LoadBalancerDescriptions[].LoadBalancerName' --output text); do
  if aws elb describe-tags --region $R --load-balancer-names "$lb" --query 'TagDescriptions[0].Tags[].Key' --output text \
       | tr '\t' '\n' | grep -qx "kubernetes.io/cluster/$C"; then
    echo "deleting classic load balancer $lb"; aws elb delete-load-balancer --region $R --load-balancer-name "$lb"
  fi
done

for ng in $(aws eks list-nodegroups --region $R --cluster-name $C --query 'nodegroups[]' --output text); do
  echo "deleting node group $ng"; aws eks delete-nodegroup --region $R --cluster-name $C --nodegroup-name "$ng" >/dev/null
done
for ng in $(aws eks list-nodegroups --region $R --cluster-name $C --query 'nodegroups[]' --output text); do
  echo "waiting for $ng to go (several minutes)"; aws eks wait nodegroup-deleted --region $R --cluster-name $C --nodegroup-name "$ng"
done
echo "deleting cluster $C"
aws eks delete-cluster --region $R --name $C >/dev/null
aws eks wait cluster-deleted --region $R --name $C
echo "argus-eks deleted"
aws ec2 describe-instances --region $R --filters "Name=tag:eks:cluster-name,Values=$C" "Name=instance-state-name,Values=pending,running" \
  --query 'length(Reservations[].Instances[])' --output text | sed 's/^/cluster nodes still running: /'

#!/usr/bin/env bash
# One-time (idempotent) setup so GitHub Actions can deploy without static AWS keys.
#
# Creates the GitHub OIDC identity provider and a role that only workflows running on the
# master branch of this repository can assume. The role can push to the argus/* ECR
# repositories and describe the EKS cluster, nothing else. Needs IAM admin credentials:
#
#   AWS_PROFILE=argus-demo ./scripts/setup_github_oidc.sh
#
# Afterwards, in .github/workflows/deploy.yml replace the static-key step with:
#   permissions: {id-token: write, contents: read}
#   - uses: aws-actions/configure-aws-credentials@v4
#     with: {role-to-assume: arn:aws:iam::<account>:role/argus-github-actions, aws-region: eu-north-1}
# and delete the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY repository secrets.
#
# The repository name must match GitHub's CURRENT canonical name (`gh repo view`), because
# that is what appears in the token's `sub` claim. Do not also trust a previous owner name:
# if that username is ever released, someone else could claim it.
#
# Helm deploys also need the role mapped into the cluster (EKS access entry) once a cluster
# exists:
#   aws eks create-access-entry --cluster-name argus-eks --principal-arn <role arn>
#   aws eks associate-access-policy --cluster-name argus-eks --principal-arn <role arn> \
#     --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
#     --access-scope type=cluster
set -euo pipefail

REPO="${REPO:-Angshuman1211/Argus-Spot_Resilient_ML_Training_Orchestrator}"
BRANCH="${BRANCH:-master}"
ROLE="${ROLE:-argus-github-actions}"
REGION="${AWS_REGION:-eu-north-1}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
PROVIDER_ARN="arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"

if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER_ARN" >/dev/null 2>&1; then
  echo "OIDC provider exists"
else
  # AWS validates GitHub's certificate chain itself; the thumbprint is required by the API but unused.
  aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 >/dev/null
  echo "created OIDC provider"
fi

TRUST="$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${PROVIDER_ARN}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        "token.actions.githubusercontent.com:sub": "repo:${REPO}:ref:refs/heads/${BRANCH}"
      }
    }
  }]
}
JSON
)"

if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "$TRUST"
  echo "updated trust policy on $ROLE"
else
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" \
    --description "GitHub Actions deploy role for ${REPO} (${BRANCH} only)" >/dev/null
  echo "created role $ROLE"
fi

POLICY="$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
    {"Sid": "EcrPush", "Effect": "Allow",
     "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
                "ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"],
     "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT}:repository/argus/*"},
    {"Sid": "EksDescribe", "Effect": "Allow", "Action": ["eks:DescribeCluster", "eks:ListClusters"], "Resource": "*"}
  ]
}
JSON
)"
aws iam put-role-policy --role-name "$ROLE" --policy-name deploy --policy-document "$POLICY"
echo "role ARN: arn:aws:iam::${ACCOUNT}:role/${ROLE}"

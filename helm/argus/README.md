# Argus — Deployment Guide

Deploys the Kubernetes operator that watches `SpotResilientJob` resources,
polls the prediction service for risk scores, and checkpoints + migrates
training pods ahead of a Spot interruption. Works against **minikube**
(free, use this for demos) or a real **EKS** cluster.

## Prerequisites

- A cluster (`minikube start`, or a live EKS cluster — see `terraform/`)
- `kubectl` pointed at that cluster
- `helm` 3.x
- Images already built and pushed to ECR (`scripts/push_operator.sh`). The chart
  derives the image, bucket, queue and IRSA role from one value, `aws.accountId`
  (`<accountId>.dkr.ecr.<region>.amazonaws.com/argus/operator:latest` by default)
- On EKS: the `argus-operator-irsa` IAM role must already exist (Terraform
  output) — the chart's ServiceAccount annotates itself with that role's ARN
  for AWS access (S3 checkpoints, SQS risk events), no static credentials.
  On minikube there's no IRSA — see `docs/problems_and_decisions.md` P-026
  for how the real operator gets AWS creds off-EKS for a demo.

## Install

```bash
helm upgrade --install argus ./helm/argus --namespace default \n  --set aws.accountId=$(aws sts get-caller-identity --query Account --output text)
```

This deploys, in one release: the `spotresilientjobs.argus.io` CRD, the
operator Deployment, its RBAC (ClusterRole watching `SpotResilientJob` +
pods/nodes), its ServiceAccount, and a metrics Service Prometheus can scrape
via pod annotations. Read the printed NOTES after install — it has the
riskThreshold caveat below and quick verification commands.

Override any `values.yaml` key with `--set`, e.g. a different image tag:
```bash
helm upgrade --install argus ./helm/argus --set image.tag=v1.2.3
```

## Deploy the prediction service

The operator calls out to this over HTTP (`operator.predictServiceUrl` in
`values.yaml`, default `http://argus-predict-service:8000`) — it's a plain
manifest, not part of this chart:

```bash
envsubst < k8s/predict-service.yaml | kubectl apply -f -
```

## Deploy a training job

```bash
envsubst < demo/spotresilientjob.yaml | kubectl apply -f -
kubectl get spotresilientjob   # shortname: srj
```

### `riskThreshold` — read this before deploying against the real model

The CRD **requires** every job to set its own `riskThreshold` — there's no
way to omit it and fall back to the chart's default in practice. Two
different scales are in play depending on which prediction service you point
`PREDICT_SERVICE_URL` at:

| Predict service | Real output range | Threshold that actually fires |
|---|---|---|
| Real model (`ml/api/app.py`, gradient boosting) | 0 – 1 (calibrated probability of a >1% price rise within 15 min) | **~0.05** (10% alarm budget for eu-north-1, chosen on the calibration slice: `docs/trace_benchmark_result.md`, `/model` endpoint) |
| Mock predictor (`demo/mock-predict.yaml`, used for the Week 7 dashboard demo) | 0.15 → 0.90 ramp | 0.65 (the historical default, still correct for this case) |

`demo/spotresilientjob.yaml` ships with `riskThreshold: 0.65` because it was
built for the mock-predictor demo. **If you point it at the real model
instead, change this to ~0.05, otherwise the predictive path will fire on every poll** — this was true of every deployment before Week 7 and went
unnoticed because the dashboard demo (which looks fine) used the mock, not
the real model. Don't copy `0.65` into a new job spec without checking which
predict service it's actually talking to.

## Observability

```bash
./scripts/monitoring.sh up      # Prometheus + Grafana into ns `monitoring`
./scripts/monitoring.sh open    # port-forward -> http://localhost:3000
./scripts/monitoring.sh status  # confirm the operator is actually being scraped
./scripts/monitoring.sh down    # tear down
```

Dashboard **"Argus — Spot Resilience"** is provisioned automatically on
Grafana startup — no manual setup. See `k8s/monitoring/README.md` for the
metric reference and `docs/A6_observability.md` for how it was built.

## Teardown

```bash
helm uninstall argus
kubectl delete -f k8s/predict-service.yaml
./scripts/monitoring.sh down
```

On EKS specifically, also tear down the node groups/control plane
(`terraform destroy` scoped to the cluster) — the EKS control plane bills
~$0.10/hr even with zero pods running. See `scripts/verify_teardown.sh` to
confirm nothing billable is left behind in any region.

## Chart contents

| File | What it deploys |
|---|---|
| `templates/crd.yaml` | `SpotResilientJob` CRD |
| `templates/deployment.yaml` | operator Deployment (metrics port + scrape annotations) |
| `templates/rbac.yaml` | ClusterRole/Binding — watch CRs, cordon/delete pods+nodes |
| `templates/serviceaccount.yaml` | ServiceAccount, IRSA-annotated for EKS |
| `templates/service.yaml` | metrics-only Service (`:8080/metrics`) |
| `templates/NOTES.txt` | printed post-install/upgrade, has the riskThreshold reminder |

Not part of this chart (separate plain manifests, see above): the prediction
service (`k8s/predict-service.yaml`) and the monitoring stack
(`k8s/monitoring/`) — kept out so a single `helm upgrade` for the operator
doesn't also churn unrelated components.

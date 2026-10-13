import os
from prometheus_client import Counter, Gauge, start_http_server

# Metrics
checkpoint_count = Counter(
    "argus_checkpoints_total", "Total number of checkpoints triggered", ["job_name"]
)
risk_score_gauge = Gauge(
    "argus_risk_score", "Risk score time series", ["job_name", "instance_type", "az"]
)
job_completion_count = Counter(
    "argus_jobs_completed_total", "Total jobs successfully completed", ["job_name"]
)

checkpoint_wait_timeouts = Counter(
    "argus_checkpoint_wait_timeouts_total",
    "Migrations started without a confirmed checkpoint flush", ["job_name"]
)

def start_metrics_server():
    port = int(os.environ.get("METRICS_PORT", 8080))
    start_http_server(port)
    print(f"Metrics server started on port {port}")

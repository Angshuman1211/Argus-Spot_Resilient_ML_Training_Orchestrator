#!/bin/bash
# Boot script for the cloud campaign runner (an EC2 machine started by scripts/start_cloud_runner.py).
# Downloads the campaign code, state and any data so far from S3, then runs one systemd service per region
# (restarted on failure, started again after a reboot), syncs results back to S3 every 2 minutes, and shuts the
# machine down (= terminates it) 10 minutes after the campaign ends.
set -x
BUCKET="$1"
B="s3://$BUCKET/campaign"
export AWS_DEFAULT_REGION=eu-north-1
D=/opt/argus
mkdir -p $D/scripts $D/ml/data/real
dnf install -y python3-pip
pip3 install -q boto3
aws s3 cp $B/code/sensor_campaign.py $D/scripts/sensor_campaign.py
aws s3 cp $B/state.json $D/state.json
aws s3 sync $B/data/ $D/ml/data/real/

read -r END PER MAX <<<"$(python3 -c "import json; s=json.load(open('$D/state.json')); print(s['end_utc'], s['per_pool'], s['max_launches'])")"
python3 -c "import json; [print(r, t) for r, t in json.load(open('$D/state.json'))['regions'].items()]" | while read -r REGION TAG; do
cat > /etc/systemd/system/argus-campaign-$REGION.service <<EOF
[Unit]
Description=argus sensor campaign $REGION
After=network-online.target
Wants=network-online.target
[Service]
WorkingDirectory=$D
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 scripts/sensor_campaign.py --region $REGION --end-utc $END --resume-tag $TAG --per-pool $PER --max-launches $MAX
StandardOutput=append:$D/ml/data/real/campaign_$TAG.log
StandardError=append:$D/ml/data/real/campaign_$TAG.err
Restart=on-failure
RestartSec=60
[Install]
WantedBy=multi-user.target
EOF
systemctl enable --now argus-campaign-$REGION.service
done

cat > /etc/systemd/system/argus-sync.service <<EOF
[Service]
Type=oneshot
Environment=AWS_DEFAULT_REGION=eu-north-1
ExecStart=/usr/bin/aws s3 sync $D/ml/data/real/ $B/data/
EOF
cat > /etc/systemd/system/argus-sync.timer <<EOF
[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
[Install]
WantedBy=timers.target
EOF
cat > $D/finish.sh <<EOF
#!/bin/bash
while [ \$(date -u +%s) -lt \$(( \$(date -u -d "$END" +%s) + 600 )) ]; do sleep 60; done
systemctl start argus-sync.service
aws s3 cp /var/log/cloud-init-output.log $B/data/cloud_runner_boot.log
shutdown -h now
EOF
chmod +x $D/finish.sh
cat > /etc/systemd/system/argus-finish.service <<EOF
[Service]
Environment=AWS_DEFAULT_REGION=eu-north-1
ExecStart=$D/finish.sh
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now argus-sync.timer argus-finish.service
echo "runner ready"

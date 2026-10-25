# Watchdog for the sensor campaigns: safe to run any number of times. It reads ml\data\real\campaign_state.json
# (written by scripts\start_campaign.ps1) and starts whatever is not running: one campaign per region, the
# keep-awake helper and the finish pipeline. Task Scheduler runs it at logon and every 10 minutes, so a reboot, a
# crash or a network drop only delays the campaign. When the finish pipeline has logged DONE it removes its own task.
#   powershell -NoProfile -WindowStyle Hidden -File scripts\resume_campaign.ps1
param([string]$TaskName = "argus-campaign-watchdog")
$root = Split-Path -Parent $PSScriptRoot
$real = Join-Path $root "ml\data\real"
$log = Join-Path $real "watchdog.log"
function Say($m) { "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss'))Z $m" | Add-Content -Path $log }
$statePath = Join-Path $real "campaign_state.json"
if (-not (Test-Path $statePath)) { exit 0 }
$s = Get-Content $statePath -Raw | ConvertFrom-Json
$env:AWS_PROFILE = $s.profile
$env:PYTHONIOENCODING = "utf-8"
$end = [datetime]::SpecifyKind([datetime]::Parse($s.end_utc), [DateTimeKind]::Utc)
$now = (Get-Date).ToUniversalTime()

# Spot quota requests: logs a change and pops a notice once when AWS answers (state in quota_watch.json)
Start-Process $s.python -WindowStyle Hidden -WorkingDirectory $root -ArgumentList "scripts\quota_watch.py"

function Running($pattern) {
    @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern }).Count -gt 0
}

# finished? the pipeline's log says DONE after the "waiting until <this end>" line
$pl = Join-Path $real "post_campaign.log"
if (Test-Path $pl) {
    $lines = @(Get-Content $pl)
    $idx = -1
    $mark = "waiting until " + $end.ToString("yyyy-MM-dd HH:mm")
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match [regex]::Escape($mark)) { $idx = $i } }
    if ($idx -ge 0 -and ($lines[$idx..($lines.Count - 1)] | Where-Object { $_ -match " DONE$" })) {
        Say "pipeline finished; removing scheduled task"
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        exit 0
    }
}

if ($now -lt $end.AddHours(1) -and -not (Running "keep_awake")) {
    Say "starting keep-awake"
    $h = [math]::Ceiling(($end - $now).TotalHours + 1)
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile","-WindowStyle","Hidden","-File","$root\scripts\keep_awake.ps1","-Hours","$h"
}
$cloud = ($s.mode -eq "cloud")  # the campaigns run on the cloud runner; this laptop only does the finish
foreach ($p in $s.regions.PSObject.Properties) {
    $region = $p.Name; $tag = $p.Value
    if (-not $cloud -and $now -lt $end.AddMinutes(-10) -and -not (Running "sensor_campaign\.py.*--region $region\b")) {
        Say "starting campaign in $region (tag $tag)"
        Start-Process $s.python -WindowStyle Hidden -WorkingDirectory $root `
            -RedirectStandardOutput "$real\campaign_$tag.log" -RedirectStandardError "$real\campaign_$tag.err" `
            -ArgumentList "scripts\sensor_campaign.py","--region",$region,"--end-utc",$s.end_utc,"--resume-tag",$tag,
                          "--per-pool","$($s.per_pool)","--max-launches","$($s.max_launches)"
    }
}
if (-not (Running "post_campaign\.py")) {
    Say "starting finish pipeline"
    $regions = ($s.regions.PSObject.Properties | ForEach-Object { $_.Name }) -join ","
    # Start-Process joins arguments with spaces, so paths such as C:\Program Files\... must carry their own quotes
    $args_ = @("scripts\post_campaign.py","--end-utc",$s.end_utc,"--regions",$regions,"--python","`"$($s.python)`"","--tectonic","`"$($s.tectonic)`"")
    if ($cloud) { $args_ += @("--s3-data", "s3://$($s.bucket)/campaign/data/") }
    Start-Process $s.python -WindowStyle Hidden -WorkingDirectory $root `
        -RedirectStandardOutput "$real\post_campaign.out" -RedirectStandardError "$real\post_campaign.err" `
        -ArgumentList $args_
}

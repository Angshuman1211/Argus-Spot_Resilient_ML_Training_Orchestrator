# Watchdog for the sensor campaign: safe to run any number of times. It starts whatever is not running
# (campaign, keep-awake, finish pipeline) and does nothing for what is. Meant to be run by Windows Task Scheduler
# at logon and every 10 minutes, so a reboot, a crash or a network drop only delays the campaign, never ends it.
# When the finish pipeline has logged DONE it removes its own scheduled task.
#   powershell -NoProfile -WindowStyle Hidden -File scripts\resume_campaign.ps1
param(
    [string]$EndUtc = "2026-10-13T22:41",
    [string]$Tag = "20261013T144104Z",
    [string]$Python = "C:\Users\rayya\AppData\Local\Temp\claude\f--Argus-boto3\8b26b80d-26be-43f5-ba5f-610ebb9d9ca7\scratchpad\gpuenv\Scripts\python.exe",
    [string]$Tectonic = "C:\Users\rayya\AppData\Local\Temp\claude\f--Argus-boto3\8b26b80d-26be-43f5-ba5f-610ebb9d9ca7\scratchpad\tectonic_bin\tectonic.exe",
    [string]$TaskName = "argus-campaign-watchdog"
)
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "ml\data\real\watchdog.log"
function Say($m) { "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss'))Z $m" | Add-Content -Path $log }
$env:AWS_PROFILE = "argus-demo"
$env:PYTHONIOENCODING = "utf-8"
$end = [datetime]::SpecifyKind([datetime]::Parse($EndUtc), [DateTimeKind]::Utc)
$now = (Get-Date).ToUniversalTime()

function Running($pattern) {
    @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern }).Count -gt 0
}

# finished? the pipeline's own log says DONE after its last "waiting until" line
$pl = Join-Path $root "ml\data\real\post_campaign.log"
if (Test-Path $pl) {
    $lines = Get-Content $pl
    $idx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match "waiting until") { $idx = $i } }
    if ($idx -ge 0 -and ($lines[($idx + 1)..($lines.Count - 1)] | Where-Object { $_ -match " DONE$" })) {
        Say "pipeline finished; removing scheduled task"
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        exit 0
    }
}

if ($now -lt $end.AddHours(1)) {
    if (-not (Running "keep_awake")) {
        Say "starting keep-awake"
        Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile","-WindowStyle","Hidden","-File","$root\scripts\keep_awake.ps1","-Hours","6"
    }
}
if ($now -lt $end -and -not (Running "sensor_campaign\.py")) {
    Say "starting campaign (resume)"
    Start-Process $Python -WindowStyle Hidden -WorkingDirectory $root -RedirectStandardOutput "$root\ml\data\real\campaign_resume.log" -RedirectStandardError "$root\ml\data\real\campaign_resume.err" `
        -ArgumentList "scripts\sensor_campaign.py","--end-utc",$EndUtc,"--resume-tag",$Tag,"--max-launches","400"
}
if (-not (Running "post_campaign\.py")) {
    Say "starting finish pipeline"
    Start-Process $Python -WindowStyle Hidden -WorkingDirectory $root -RedirectStandardOutput "$root\ml\data\real\post_campaign.out" -RedirectStandardError "$root\ml\data\real\post_campaign.err" `
        -ArgumentList "scripts\post_campaign.py","--end-utc",$EndUtc,"--python",$Python,"--tectonic",$Tectonic,"--delete-root-key"
}

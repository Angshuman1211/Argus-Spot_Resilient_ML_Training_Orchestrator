# Starts a multi-region reclaim-sensor campaign: writes ml\data\real\campaign_state.json and runs the watchdog
# once, which launches one campaign per region, keep-awake, and the finish pipeline. With -RegisterTask it also
# registers the watchdog in Task Scheduler (at logon and every 10 minutes) so the run survives a restart.
#   powershell -File scripts\start_campaign.ps1 -Hours 12 -RegisterTask
param(
    [string[]]$Regions = @("eu-north-1", "us-east-1", "us-east-2", "ap-southeast-1"),
    [double]$Hours = 12,
    [int]$PerPool = 2,
    [int]$MaxLaunches = 600,
    [string]$AwsProfile = "argus-sensor",
    [string]$Python = "C:\Users\rayya\AppData\Local\Temp\claude\f--Argus-boto3\8b26b80d-26be-43f5-ba5f-610ebb9d9ca7\scratchpad\gpuenv\Scripts\python.exe",
    [string]$Tectonic = "C:\Users\rayya\AppData\Local\Temp\claude\f--Argus-boto3\8b26b80d-26be-43f5-ba5f-610ebb9d9ca7\scratchpad\tectonic_bin\tectonic.exe",
    [switch]$RegisterTask
)
$root = Split-Path -Parent $PSScriptRoot
$now = (Get-Date).ToUniversalTime()
$stamp = $now.ToString("yyyyMMddTHHmmssZ")
$tags = [ordered]@{}
foreach ($r in $Regions) { $tags[$r] = "${r}_$stamp" }
$state = [ordered]@{
    end_utc = $now.AddHours($Hours).ToString("yyyy-MM-ddTHH:mm"); profile = $AwsProfile; python = $Python
    tectonic = $Tectonic; per_pool = $PerPool; max_launches = $MaxLaunches; regions = $tags
}
$state | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root "ml\data\real\campaign_state.json")
"campaign until $($state.end_utc)Z in $($Regions -join ', ')"

if ($RegisterTask) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -File $root\scripts\resume_campaign.ps1"
    $t1 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Hours ($Hours + 3))
    $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -Hidden -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
    Register-ScheduledTask -TaskName "argus-campaign-watchdog" -Action $action -Trigger @($t1, $t2) -Settings $set -Force | Out-Null
    "watchdog task registered"
}
& powershell -NoProfile -File (Join-Path $root "scripts\resume_campaign.ps1")
"started; watch ml\data\real\watchdog.log and ml\data\real\campaign_<region>_$stamp.log"

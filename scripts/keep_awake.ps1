# Keeps Windows from going to sleep while a long background job (the sensor campaign) runs.
# Not persistent: it holds the request only while this process is alive, and stops when it exits.
#   powershell -WindowStyle Hidden -File scripts\keep_awake.ps1 -Hours 9
param([double]$Hours = 9)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$end = (Get-Date).AddHours($Hours)
while ((Get-Date) -lt $end) {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    Start-Sleep -Seconds 60
}
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)

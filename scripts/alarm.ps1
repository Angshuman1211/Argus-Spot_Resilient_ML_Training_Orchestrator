# Alarm when the campaign's finish pipeline is done: a Windows notification, an alarm sound for about a minute,
# and a message box that stays until dismissed.
#   powershell -File scripts\alarm.ps1 -Message "test" -Seconds 3
param([string]$Message = "Argus campaign finished", [int]$Seconds = 60)

try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $t = $xml.GetElementsByTagName("text")
    [void]$t.Item(0).AppendChild($xml.CreateTextNode("Argus"))
    [void]$t.Item(1).AppendChild($xml.CreateTextNode($Message))
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
} catch {}

# the sound runs in the background so the message box can appear at the same time
$sound = Start-Job -ArgumentList $Seconds -ScriptBlock {
    param($s)
    $p = New-Object System.Media.SoundPlayer "C:\Windows\Media\Alarm01.wav"
    $end = (Get-Date).AddSeconds($s)
    while ((Get-Date) -lt $end) { try { $p.PlaySync() } catch { [console]::beep(880, 700); Start-Sleep -Milliseconds 300 } }
}
Add-Type -AssemblyName PresentationFramework
[void][System.Windows.MessageBox]::Show($Message, "Argus campaign", "OK", "Information")
Stop-Job $sound -ErrorAction SilentlyContinue; Remove-Job $sound -Force -ErrorAction SilentlyContinue

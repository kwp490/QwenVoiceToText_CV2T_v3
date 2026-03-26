$scripts = @(
    (Join-Path $PSScriptRoot 'Install-CV2T-Source.ps1'),
    (Join-Path $PSScriptRoot 'Install-CV2T-Bin.ps1')
)
$totalErrors = 0
foreach ($s in $scripts) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($s, [ref]$tokens, [ref]$errors) | Out-Null
    $name = Split-Path -Leaf $s
    Write-Host "$name : $($errors.Count) parse errors"
    foreach ($e in $errors) { Write-Host "  $($e.Message) at line $($e.Extent.StartLineNumber)" }
    $totalErrors += $errors.Count
}
if ($totalErrors -gt 0) { exit 1 }

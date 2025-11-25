[CmdletBinding()]
param([switch]$SkipTests, [switch]$SkipDocs)

# Core checks (always run)
$CoreSteps = @(
    "black .",
    "isort .",
    "flake8 ."
)

# Optional checks
$TestSteps = @("pytest")
$DocSteps = @(
    "pyreverse -o svg -d .\docs .\ai_toolkit\",
    "cd docs; .\make.bat clean; .\make.bat html; cd .."
)

# Other steps (always run)
$OtherSteps = @(
    "pipreqs . --force --mode gt",
    @{
        Command = '$env:PYTHONIOENCODING="utf-8"; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; project2md process --output=.idea/summary.md'
        Display = 'project2md process --output=.idea/summary.md'
    }
)

# Build final step list
$AllSteps = $CoreSteps
if (-not $SkipTests) { $AllSteps += $TestSteps }
if (-not $SkipDocs) { $AllSteps += $DocSteps }
$AllSteps += $OtherSteps

Write-Host "Running $($AllSteps.Count) pre-commit checks..." -ForegroundColor Cyan
if ($SkipTests) { Write-Host "   Skipping tests" -ForegroundColor Yellow }
if ($SkipDocs) { Write-Host "   Skipping docs" -ForegroundColor Yellow }
Write-Host ""

# Execute steps
$Failed = 0
for ($i = 0; $i -lt $AllSteps.Count; $i++) {
    $Step = $AllSteps[$i]

    if ($Step -is [hashtable]) {
        $Command = $Step.Command
        $DisplayStep = $Step.Display
    } else {
        $Command = $Step
        $DisplayStep = $Step
    }

    Write-Host "[$($i+1)/$($AllSteps.Count)] " -NoNewline -ForegroundColor Blue
    Write-Host $DisplayStep -ForegroundColor White
    Invoke-Expression $Command 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "      FAILED" -ForegroundColor Red
        $Failed++
    }
}

# Summary
Write-Host ""
if ($Failed -eq 0) {
    Write-Host "All $($AllSteps.Count) checks passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$Failed of $($AllSteps.Count) checks failed!" -ForegroundColor Red
    exit 1
}

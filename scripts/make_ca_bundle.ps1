# Only needed on machines behind a TLS-intercepting proxy (corporate MITM),
# where pip / huggingface_hub / groq fail with CERTIFICATE_VERIFY_FAILED.
#
# Merges the Windows root+intermediate CA stores into certifi's bundle and
# writes certs/ca-bundle.pem, which backend/app/config.py picks up automatically.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_ca_bundle.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

$certifiPath = & $python -c "import certifi; print(certifi.where())"
$bundle = Get-Content $certifiPath -Raw

$stores = "Cert:\LocalMachine\Root", "Cert:\CurrentUser\Root", "Cert:\LocalMachine\CA", "Cert:\CurrentUser\CA"
foreach ($cert in (Get-ChildItem $stores -ErrorAction SilentlyContinue)) {
    $b64 = [Convert]::ToBase64String($cert.RawData, 'InsertLineBreaks')
    $bundle += "`n# $($cert.Subject)`n-----BEGIN CERTIFICATE-----`n$b64`n-----END CERTIFICATE-----`n"
}

$outDir = Join-Path $root "certs"
New-Item -ItemType Directory -Force $outDir | Out-Null
Set-Content -Path (Join-Path $outDir "ca-bundle.pem") -Value $bundle -Encoding ascii
Write-Host "Wrote $outDir\ca-bundle.pem"

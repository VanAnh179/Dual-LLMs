# =============================================================
# Setup script - runs on your Windows machine
# Purpose:
#   1. Rename old repo (D11.x/D12.x) for clarity, no content change.
#   2. Create new repo "cospec-ssb" for SSB track (S01, S02, ...).
#   3. Copy reusable assets (raw GSM8K data, src/ utils, D11.0 adapters,
#      train_alternating_lora.py script for fallback retrain) to the new repo.
#      This is an independent COPY - edits in new repo will NOT affect old repo.
# Usage: Open PowerShell at "D:\Program\Dual LLMs\" and run:
#        powershell -ExecutionPolicy Bypass -File setup_cospec_ssb_repo.ps1
# =============================================================

$root    = "D:\Program\Dual LLMs"
$oldRepoOriginalName = "$root\gsm8k-dual-agent-finetune"
$oldRepo = "$root\gsm8k-text-collab-baselines"
$newRepo = "$root\cospec-ssb"

# --- 1. Rename old repo ---
if (Test-Path $oldRepoOriginalName) {
    Rename-Item -Path $oldRepoOriginalName -NewName "gsm8k-text-collab-baselines"
    Write-Host "[OK] Renamed: gsm8k-dual-agent-finetune -> gsm8k-text-collab-baselines"
} elseif (Test-Path $oldRepo) {
    Write-Host "[SKIP] Old repo already renamed to gsm8k-text-collab-baselines, skipping rename."
} else {
    Write-Host "[ERROR] Could not find '$oldRepoOriginalName' or '$oldRepo'. Please check the path." -ForegroundColor Red
    exit 1
}

# --- 2. Create new repo directory structure ---
$dirs = @(
    "$newRepo\src",
    "$newRepo\scripts",
    "$newRepo\configs",
    "$newRepo\data\raw",
    "$newRepo\data\filtered",
    "$newRepo\data\train",
    "$newRepo\outputs",
    "$newRepo\outputs\imported_d11_adapters",
    "$newRepo\notes"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}
Write-Host "[OK] Created directory structure at $newRepo"

# --- 3. Copy raw GSM8K data (shared, unchanged) ---
Copy-Item "$oldRepo\data\raw\*.jsonl" "$newRepo\data\raw\" -Force
Write-Host "[OK] Copied data/raw/*.jsonl"

# --- 4. Copy reusable src/ modules (independent copy) ---
$reusableSrc = @(
    "data_utils.py",
    "generation.py",
    "evaluation.py",
    "answer_extraction.py",
    "prompts.py",
    "training.py"
)
foreach ($f in $reusableSrc) {
    $srcPath = "$oldRepo\src\$f"
    if (Test-Path $srcPath) {
        Copy-Item $srcPath "$newRepo\src\$f" -Force
        Write-Host "[OK] Copied src/$f"
    } else {
        Write-Host "[SKIP] Could not find src/$f in old repo - check if needed." -ForegroundColor Yellow
    }
}

# --- 5. Copy train_alternating_lora.py (fallback retrain if adapter fails) ---
$trainScript = "$oldRepo\scripts\train_alternating_lora.py"
if (Test-Path $trainScript) {
    Copy-Item $trainScript "$newRepo\scripts\train_alternating_lora.py" -Force
    Write-Host "[OK] Copied scripts/train_alternating_lora.py (used for fallback retrain of D11.0 if adapter fails)"
} else {
    Write-Host "[SKIP] Could not find scripts/train_alternating_lora.py - check file name in old repo." -ForegroundColor Yellow
}

# --- 6. Copy D11.0 adapters to "imported" directory ---
$adapterA = "$oldRepo\outputs\adapters\agent_A_round_1"
$adapterB = "$oldRepo\outputs\adapters\agent_B_round_1"

if (Test-Path $adapterA) {
    Copy-Item $adapterA "$newRepo\outputs\imported_d11_adapters\agent_A_round_1" -Recurse -Force
    $sizeA = (Get-ChildItem "$newRepo\outputs\imported_d11_adapters\agent_A_round_1" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("[OK] Copied adapter A ({0:N2} MB)" -f $sizeA)
    if ($sizeA -lt 5) {
        Write-Host "[WARNING] Adapter A is only ~$([math]::Round($sizeA,2)) MB - might be truncated as noted in D12_0_voting_report.md. Need to retrain using scripts/train_alternating_lora.py in new repo if loading fails." -ForegroundColor Yellow
    }
} else {
    Write-Host "[SKIP] Could not find adapter A at '$adapterA' - will need to retrain in new repo." -ForegroundColor Yellow
}

if (Test-Path $adapterB) {
    Copy-Item $adapterB "$newRepo\outputs\imported_d11_adapters\agent_B_round_1" -Recurse -Force
    $sizeB = (Get-ChildItem "$newRepo\outputs\imported_d11_adapters\agent_B_round_1" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("[OK] Copied adapter B ({0:N2} MB)" -f $sizeB)
    if ($sizeB -lt 5) {
        Write-Host "[WARNING] Adapter B is only ~$([math]::Round($sizeB,2)) MB - might be truncated. Need to retrain if loading fails." -ForegroundColor Yellow
    }
} else {
    Write-Host "[SKIP] Could not find adapter B at '$adapterB' - will need to retrain in new repo." -ForegroundColor Yellow
}

# --- 7. Copy reference documents (optional but helpful) ---
$refDocs = @("README.md", "RUN.md", "requirements.txt")
foreach ($f in $refDocs) {
    $p = "$oldRepo\$f"
    if (Test-Path $p) {
        Copy-Item $p "$newRepo\$f" -Force
        Write-Host "[OK] Copied $f (reference)"
    }
}

Write-Host ""
Write-Host "===================================================="
Write-Host "DONE."
Write-Host "Old repo (D11.x/D12.x, keep as-is, do not modify): $oldRepo"
Write-Host "New repo (SSB track, work from here):              $newRepo"
Write-Host "===================================================="

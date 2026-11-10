# WhisperLiveKit live server - multilingual (faster-whisper)
# Run from YOUR OWN PowerShell:   cd C:\Users\manis\Downloads\whisper ; .\start_server.ps1
#
# Device/model are auto-detected: CUDA GPU if one is present, otherwise CPU.
# On CPU the defaults are deliberately smaller, because large-v3 runs at ~0.5x
# realtime there and can never keep up with a live stream.
#
# Examples:
#   .\start_server.ps1                         (auto-detect GPU/CPU)
#   .\start_server.ps1 -Language hi            (force Hindi - recommended for Hindi speech)
#   .\start_server.ps1 -Model large-v3-turbo   (faster than large-v3, a bit less accurate)
#   .\start_server.ps1 -Device cuda -Compute float16
#   .\start_server.ps1 -Model tiny             (lowest lag on CPU)
param(
    [string]$Model = "",
    [string]$Language = "auto",
    [string]$Compute = "",          # default: int8_float16 on GPU, int8 on CPU
    [string]$Device = "auto",       # auto | cuda | cpu
    [int]$CpuThreads = 0,           # CPU mode only; 0 = library default
    [double]$MinChunkSize = 0,      # 0 = pick per device (0.1 GPU, 1.0 CPU)
    [int]$BufferTrimmingSec = 0,    # 0 = pick per device (15 GPU, 10 CPU)
    [int]$Port = 8000,
    [switch]$Diarization,
    [string[]]$ExtraArgs = @()      # anything else passed straight to the server
)

$root = $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "venv not found at $py"
    Write-Host "Create it first:  py -3.12 -m venv venv"
    exit 1
}

# Make pip-installed CUDA libs (cuBLAS / cuDNN) visible to ctranslate2
Get-ChildItem (Join-Path $root "venv\Lib\site-packages\nvidia") -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $bin = Join-Path $_.FullName "bin"
    if (Test-Path $bin) { $env:PATH = "$bin;$env:PATH" }
}

# Auto-detect CUDA. ctranslate2 is what actually runs the model, so ask it.
if ($Device -eq "auto") {
    $n = & $py -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())" 2>$null
    if ($LASTEXITCODE -eq 0 -and [int]$n -gt 0) {
        $Device = "cuda"
        Write-Host "CUDA device detected ($n). Using GPU."
    } else {
        $Device = "cpu"
        Write-Host "No CUDA device detected. Using CPU."
    }
}

$onCpu = ($Device -eq "cpu")

# Model default depends on device: large-v3 is unusable live on CPU.
if (-not $Model) {
    $Model = if ($onCpu) { "base" } else { "large-v3" }
}

if (-not $Compute) { $Compute = if ($onCpu) { "int8" } else { "int8_float16" } }
if ($onCpu -and $Compute -like "*float16*") {
    Write-Host "float16 CPU par support nahi hota, int8 use kar raha hoon."
    $Compute = "int8"
}

# WhisperLiveKit defaults to --min-chunk-size 0.1, i.e. it attempts inference ~10x
# a second. On CPU a single pass takes about a second, so work piles up and the
# transcript falls minutes behind. Ask for less frequent, larger passes instead.
if ($MinChunkSize -le 0) { $MinChunkSize = if ($onCpu) { 1.0 } else { 0.1 } }
if ($BufferTrimmingSec -le 0) { $BufferTrimmingSec = if ($onCpu) { 10 } else { 15 } }

$env:WLK_DEVICE = $Device
$env:WLK_COMPUTE = $Compute
$env:WLK_CPU_THREADS = $CpuThreads

$a = @("--model", $Model, "--language", $Language, "--backend", "faster-whisper",
       "--backend-policy", "localagreement", "--port", $Port, "--pcm-input",
       "--min-chunk-size", $MinChunkSize, "--buffer_trimming_sec", $BufferTrimmingSec)
if ($Diarization) { $a += "--diarization" }
if ($ExtraArgs) { $a += $ExtraArgs }

Write-Host "Starting WhisperLiveKit: model=$Model language=$Language device=$Device compute=$Compute port=$Port min-chunk=$MinChunkSize"
Write-Host "Pehli baar model download hoga (base ~145 MB, large-v3 ~3 GB). Phir browser: http://localhost:$Port`n"

& $py (Join-Path $root "run_wlk.py") @a

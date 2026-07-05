param(
    [string]$VideoList = $env:VIDEO_LIST,
    [string]$VideoDirs = $env:VIDEO_DIRS,
    [string]$ImageDirs = $env:IMAGE_DIRS,
    [string]$BorderImageRoot = $env:BORDER_IMAGE_ROOT,
    [string]$FullRoot = $env:FULL_ROOT,
    [string]$RunTag = $env:RUN_TAG
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
if ([string]::IsNullOrWhiteSpace($env:PYTHONIOENCODING)) {
    $env:PYTHONIOENCODING = "utf-8"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PathsPy = Join-Path $ScriptDir "cv_paths.py"

function Get-EnvOrDefault {
    param([string]$Name, [string]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

function Test-Enabled {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
    return @("1", "true", "yes", "y", "on") -contains $Value.ToLowerInvariant()
}

function Split-ConfiguredList {
    param([string]$Raw)
    if ([string]::IsNullOrWhiteSpace($Raw)) { return @() }
    return $Raw -split "[;,]" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

function Write-Utf8NoBomLines {
    param([string]$Path, [string[]]$Lines)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Path, $Lines, $encoding)
}

function Get-PathValue {
    param([string]$Key)
    $value = & $PythonBin $PathsPy $Key
    if ($LASTEXITCODE -ne 0) {
        throw "cv_paths.py failed for key: $Key"
    }
    return ($value | Select-Object -Last 1).Trim()
}

if ([string]::IsNullOrWhiteSpace($env:PYTHON_BIN)) {
    $parentVenvPython = Join-Path (Split-Path -Parent $RepoRoot) ".venv\Scripts\python.exe"
    $repoVenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $parentVenvPython) {
        $PythonBin = $parentVenvPython
    } elseif (Test-Path -LiteralPath $repoVenvPython) {
        $PythonBin = $repoVenvPython
    } else {
        $PythonBin = "python"
    }
} else {
    $PythonBin = $env:PYTHON_BIN
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
}
if ([string]::IsNullOrWhiteSpace($FullRoot)) {
    $FullRoot = Join-Path $ScriptDir "full_runs\$RunTag"
}

$TrackingRoot = Get-EnvOrDefault "TRACKING_ROOT" (Join-Path $FullRoot "tracking")
$CountRoot = Get-EnvOrDefault "COUNT_ROOT" (Join-Path $FullRoot "count")
$ImageCountRoot = Get-EnvOrDefault "IMAGE_COUNT_ROOT" (Join-Path $FullRoot "image_count")
$PipelineLogDir = Join-Path $FullRoot "logs"

$ModelPath = Get-EnvOrDefault "MODEL_PATH" (Get-PathValue "model_pt_path")
$TimeLimitJson = Get-EnvOrDefault "TIME_LIMIT_JSON" (Get-PathValue "time_limit_json_path")
$SourceJson = Get-EnvOrDefault "SOURCE_JSON" (Get-PathValue "source_json_path")
$GateLineJson = Get-EnvOrDefault "GATE_LINE_JSON" (Get-PathValue "gate_line_json_path")
$ReidModelPath = Get-EnvOrDefault "REID_MODEL_PATH" (Join-Path $ScriptDir "models\openvino\public\vehicle-reid-0001\FP32\vehicle-reid-0001.xml")

# Conservative defaults for this Windows laptop: single RTX 4080 Laptop GPU, 12GB VRAM, 32GB RAM.
$GpuIds = Get-EnvOrDefault "GPU_IDS" "0"
$BatchSize = Get-EnvOrDefault "BATCH_SIZE" "128"
$TrackWorkers = Get-EnvOrDefault "TRACK_WORKERS" "14"
$CountProcesses = Get-EnvOrDefault "COUNT_PROCESSES" "18"
$TrackerBackend = Get-EnvOrDefault "TRACKER_BACKEND" "bytetrack"
$NvdecDecoderMode = Get-EnvOrDefault "NVDEC_DECODER_MODE" "demux"
$SegmentMode = Get-EnvOrDefault "SEGMENT_MODE" "hourly-from-start"
$SegmentSeconds = Get-EnvOrDefault "SEGMENT_SECONDS" "3600"
$ExpectedVideoSeconds = Get-EnvOrDefault "EXPECTED_VIDEO_SECONDS" "68100"

if ([string]::IsNullOrWhiteSpace($VideoDirs)) {
    $VideoDirs = Get-EnvOrDefault "VIDEO_DIRS" (Get-PathValue "crawler_videos_dir")
}
$VideoExtensions = Get-EnvOrDefault "VIDEO_EXTENSIONS" "ts"

$RunDetection = Get-EnvOrDefault "RUN_DETECTION" "1"
$RunGateCount = Get-EnvOrDefault "RUN_GATE_COUNT" "1"
$RunImageCount = Get-EnvOrDefault "RUN_IMAGE_COUNT" "1"
$ExtraArgs = Get-EnvOrDefault "EXTRA_ARGS" ""
$CountExtraArgs = Get-EnvOrDefault "COUNT_EXTRA_ARGS" ""

if ([string]::IsNullOrWhiteSpace($ImageDirs)) {
    $ImageDirs = Get-EnvOrDefault "IMAGE_DIRS" ""
}
$ImageDir1 = Get-EnvOrDefault "IMAGE_DIR_1" ""
$ImageDir2 = Get-EnvOrDefault "IMAGE_DIR_2" ""
if ([string]::IsNullOrWhiteSpace($BorderImageRoot)) {
    $BorderImageRoot = Get-EnvOrDefault "BORDER_IMAGE_ROOT" ""
}
$ImageConf = Get-EnvOrDefault "IMAGE_CONF" "0.4"
$ImageOpenvinoDevice = Get-EnvOrDefault "IMAGE_OPENVINO_DEVICE" "AUTO"
$ImageReidThresh = Get-EnvOrDefault "IMAGE_REID_THRESH" "0.50"
$ImageMaxDist = Get-EnvOrDefault "IMAGE_MAX_DIST" "300"
$ImageLookback = Get-EnvOrDefault "IMAGE_LOOKBACK" "3"
$ImageZoneYMinOverride = [Environment]::GetEnvironmentVariable("IMAGE_ZONE_Y_MIN")
$ImageZoneYMaxOverride = [Environment]::GetEnvironmentVariable("IMAGE_ZONE_Y_MAX")
$ImageDefaultZoneYMin = Get-EnvOrDefault "IMAGE_DEFAULT_ZONE_Y_MIN" "0.10"
$ImageDefaultZoneYMax = Get-EnvOrDefault "IMAGE_DEFAULT_ZONE_Y_MAX" "0.80"
$Image1ZoneYMin = Get-EnvOrDefault "IMAGE1_ZONE_Y_MIN" "0.25"
$Image1ZoneYMax = Get-EnvOrDefault "IMAGE1_ZONE_Y_MAX" "0.99"
$Image5ZoneYMin = Get-EnvOrDefault "IMAGE5_ZONE_Y_MIN" "0.01"
$Image5ZoneYMax = Get-EnvOrDefault "IMAGE5_ZONE_Y_MAX" "0.45"
$ImageExtraArgs = Get-EnvOrDefault "IMAGE_EXTRA_ARGS" ""

$RunRoot = Join-Path $TrackingRoot "runs\$RunTag"
$ListDir = Join-Path $RunRoot "lists"
$WorkerLogDir = Join-Path $RunRoot "logs"
$MergedManifest = Join-Path $TrackingRoot "segment_manifest.csv"

function Assert-RequiredFiles {
    $missing = $false
    $imageDirsConfigured = -not [string]::IsNullOrWhiteSpace("$ImageDirs$ImageDir1$ImageDir2$BorderImageRoot")

    if ((Test-Enabled $RunDetection) -or ((Test-Enabled $RunImageCount) -and $imageDirsConfigured)) {
        if (-not (Test-Path -LiteralPath $ModelPath)) {
            Write-Error "Required model file not found: $ModelPath"
            $missing = $true
        }
    }
    if ((Test-Enabled $RunDetection) -and ($SegmentMode.ToLowerInvariant() -eq "time-limit") -and -not (Test-Path -LiteralPath $TimeLimitJson)) {
        Write-Error "Required time_limit json not found: $TimeLimitJson"
        $missing = $true
    }
    if ((Test-Enabled $RunGateCount) -and -not (Test-Path -LiteralPath $SourceJson)) {
        Write-Error "Required source json not found: $SourceJson"
        $missing = $true
    }
    if ((Test-Enabled $RunImageCount) -and $imageDirsConfigured -and -not (Test-Path -LiteralPath $ReidModelPath)) {
        Write-Error "OpenVINO ReID model not found: $ReidModelPath"
        $missing = $true
    }
    if ($missing) { exit 1 }
}

function Show-Config {
    Write-Host "Run tag             : $RunTag"
    Write-Host "Repo root           : $RepoRoot"
    Write-Host "Python              : $PythonBin"
    Write-Host "Full root           : $FullRoot"
    Write-Host "Tracking root       : $TrackingRoot"
    Write-Host "Count root          : $CountRoot"
    Write-Host "Image count root    : $ImageCountRoot"
    Write-Host "Model path          : $ModelPath"
    Write-Host "Time limit json     : $TimeLimitJson"
    Write-Host "Source json         : $SourceJson"
    Write-Host "Gate line json      : $GateLineJson"
    Write-Host "GPU ids             : $GpuIds"
    Write-Host "Video extensions    : $VideoExtensions"
    Write-Host "Tracker backend     : $TrackerBackend"
    Write-Host "NVDEC decoder mode  : $NvdecDecoderMode"
    Write-Host "Segment mode        : $SegmentMode"
    Write-Host "Segment seconds     : $SegmentSeconds"
    Write-Host "Expected video sec  : $ExpectedVideoSeconds"
    Write-Host "BATCH_SIZE          : $BatchSize"
    Write-Host "TRACK_WORKERS       : $TrackWorkers"
    Write-Host "COUNT_PROCESSES     : $CountProcesses"
    Write-Host "RUN_DETECTION       : $RunDetection"
    Write-Host "RUN_GATE_COUNT      : $RunGateCount"
    Write-Host "RUN_IMAGE_COUNT     : $RunImageCount"
    Write-Host "Image zone defaults : image1=$Image1ZoneYMin..$Image1ZoneYMax, image5=$Image5ZoneYMin..$Image5ZoneYMax, other=$ImageDefaultZoneYMin..$ImageDefaultZoneYMax"
}

function Build-VideoList {
    New-Item -ItemType Directory -Force -Path $ListDir | Out-Null
    if (-not [string]::IsNullOrWhiteSpace($VideoList)) {
        if (-not (Test-Path -LiteralPath $VideoList)) {
            throw "Video list not found: $VideoList"
        }
        return (Resolve-Path -LiteralPath $VideoList).Path
    }

    $generated = Join-Path $ListDir "all_videos.txt"
    $videoPaths = New-Object System.Collections.Generic.List[string]
    $dirs = Split-ConfiguredList $VideoDirs
    $exts = Split-ConfiguredList $VideoExtensions | ForEach-Object { $_.TrimStart(".") }

    if ($dirs.Count -eq 0) { throw "No VIDEO_LIST provided and VIDEO_DIRS is empty." }
    if ($exts.Count -eq 0) { throw "VIDEO_EXTENSIONS is empty." }

    foreach ($dir in $dirs) {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) {
            Write-Warning "Video directory not found, skip: $dir"
            continue
        }
        foreach ($ext in $exts) {
            Get-ChildItem -LiteralPath $dir -Recurse -File -Filter "*.$ext" |
                ForEach-Object { $videoPaths.Add($_.FullName) }
        }
    }

    $unique = $videoPaths | Sort-Object -Unique
    if (-not $unique -or $unique.Count -eq 0) {
        throw "No videos found. VIDEO_DIRS=$VideoDirs, VIDEO_EXTENSIONS=$VideoExtensions"
    }
    Write-Utf8NoBomLines -Path $generated -Lines ([string[]]$unique)
    return $generated
}

function Split-VideoListByGpu {
    param([string]$VideoListPath)
    $gpuList = Split-ConfiguredList $GpuIds
    if ($gpuList.Count -eq 0) { throw "GPU_IDS is empty." }

    $gpuFiles = @()
    for ($i = 0; $i -lt $gpuList.Count; $i++) {
        $gpuFiles += [pscustomobject]@{
            Index = $i
            GpuId = $gpuList[$i]
            ListPath = Join-Path $ListDir "gpu${i}_videos.txt"
            ManifestPath = Join-Path $RunRoot "segment_manifest_gpu${i}.csv"
            LogPath = Join-Path $WorkerLogDir "gpu${i}.log"
        }
        Write-Utf8NoBomLines -Path $gpuFiles[$i].ListPath -Lines @()
    }

    $lines = Get-Content -LiteralPath $VideoListPath | Where-Object {
        $line = $_.Trim()
        $line -and -not $line.StartsWith("#")
    }
    if (-not $lines -or $lines.Count -eq 0) {
        throw "Video list has no runnable entries: $VideoListPath"
    }

    $bucketLines = @{}
    for ($i = 0; $i -lt $gpuFiles.Count; $i++) {
        $bucketLines[$i] = New-Object System.Collections.Generic.List[string]
    }
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $bucketLines[$i % $gpuFiles.Count].Add($lines[$i])
    }
    foreach ($file in $gpuFiles) {
        Write-Utf8NoBomLines -Path $file.ListPath -Lines ([string[]]$bucketLines[$file.Index])
    }

    Write-Host "Video list          : $VideoListPath"
    Write-Host "Video count         : $($lines.Count)"
    foreach ($file in $gpuFiles) {
        Write-Host "GPU $($file.GpuId) videos      : $($bucketLines[$file.Index].Count)"
        Write-Host "GPU $($file.GpuId) log         : $($file.LogPath)"
    }
    return $gpuFiles
}

function Invoke-NativeLogged {
    param([string]$LogPath, [string]$FilePath, [string[]]$Arguments)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $FilePath @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath |
            ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. Log: $LogPath"
    }
}

function Run-DetectionTracking {
    New-Item -ItemType Directory -Force -Path $TrackingRoot, $CountRoot, $PipelineLogDir, $WorkerLogDir, $ListDir | Out-Null
    $videoListPath = Build-VideoList
    $gpuFiles = Split-VideoListByGpu -VideoListPath $videoListPath

    foreach ($file in $gpuFiles) {
        $content = Get-Content -LiteralPath $file.ListPath
        if (-not $content -or $content.Count -eq 0) {
            Write-Host "GPU $($file.GpuId) list is empty, skip."
            continue
        }

        Write-Host "Starting worker $($file.Index) on CUDA_VISIBLE_DEVICES=$($file.GpuId)..."
        $env:CUDA_VISIBLE_DEVICES = "$($file.GpuId)"
        $env:OMP_NUM_THREADS = "1"
        $env:MKL_NUM_THREADS = "1"
        $env:OPENBLAS_NUM_THREADS = "1"
        $env:NUMEXPR_NUM_THREADS = "1"
        $env:BATCH_SIZE = "$BatchSize"
        $env:TRACK_WORKERS = "$TrackWorkers"
        $env:TRACKER_BACKEND = "$TrackerBackend"
        $env:NVDEC_DECODER_MODE = "$NvdecDecoderMode"

        $args = @(
            (Join-Path $ScriptDir "run_peak_hours.py"),
            "--video-list", $file.ListPath,
            "--time-limit-json", $TimeLimitJson,
            "--segment-mode", $SegmentMode,
            "--segment-seconds", $SegmentSeconds,
            "--expected-video-seconds", $ExpectedVideoSeconds,
            "--model-path", $ModelPath,
            "--tracking-root", $TrackingRoot,
            "--manifest-path", $file.ManifestPath
        )
        if (-not [string]::IsNullOrWhiteSpace($ExtraArgs)) {
            $args += (Split-ConfiguredList $ExtraArgs)
        }
        Invoke-NativeLogged -LogPath $file.LogPath -FilePath $PythonBin -Arguments $args
    }
    return $gpuFiles
}

function Merge-Manifests {
    param($GpuFiles)
    New-Item -ItemType Directory -Force -Path $TrackingRoot | Out-Null
    $header = @(
        "video_name", "cam_key", "segment_name", "video_path", "start_time", "end_time",
        "start_frame", "end_frame", "duration_sec", "is_tail", "status"
    )
    $seen = @{}
    $rows = New-Object System.Collections.Generic.List[object]

    foreach ($file in $GpuFiles) {
        if (-not (Test-Path -LiteralPath $file.ManifestPath)) {
            Write-Warning "Manifest source missing, skip: $($file.ManifestPath)"
            continue
        }
        Import-Csv -LiteralPath $file.ManifestPath | ForEach-Object {
            $key = "$($_.segment_name)|$($_.video_path)|$($_.start_frame)|$($_.end_frame)"
            if ($seen.ContainsKey($key)) { return }
            $seen[$key] = $true
            $row = [ordered]@{}
            foreach ($name in $header) {
                $row[$name] = $_.$name
            }
            $rows.Add([pscustomobject]$row)
        }
    }

    $rows |
        Sort-Object video_name, @{ Expression = { [int]($_.start_frame -as [int]) } }, segment_name |
        Export-Csv -LiteralPath $MergedManifest -NoTypeInformation -Encoding UTF8
    Write-Host "Merged manifest written: $MergedManifest ($($rows.Count) rows)"
}

function Run-GateCount {
    New-Item -ItemType Directory -Force -Path $CountRoot | Out-Null
    $args = @(
        (Join-Path $ScriptDir "VechilCountCPU.py"),
        "--csv-root", $TrackingRoot,
        "--count-root", $CountRoot,
        "--manifest-path", $MergedManifest,
        "--source-json", $SourceJson,
        "--gate-line-json", $GateLineJson,
        "--processes", "$CountProcesses"
    )
    if (-not [string]::IsNullOrWhiteSpace($CountExtraArgs)) {
        $args += (Split-ConfiguredList $CountExtraArgs)
    }
    Invoke-NativeLogged -LogPath (Join-Path $PipelineLogDir "gate_count.log") -FilePath $PythonBin -Arguments $args
}

function Resolve-ImageDirs {
    if ([string]::IsNullOrWhiteSpace($ImageDirs) -and (-not [string]::IsNullOrWhiteSpace($ImageDir1) -or -not [string]::IsNullOrWhiteSpace($ImageDir2))) {
        $ImageDirs = "$ImageDir1;$ImageDir2"
    }
    if ([string]::IsNullOrWhiteSpace($ImageDirs) -and -not [string]::IsNullOrWhiteSpace($BorderImageRoot)) {
        $ImageDirs = "$(Join-Path $BorderImageRoot "image1");$(Join-Path $BorderImageRoot "image5")"
    }
    return Split-ConfiguredList $ImageDirs
}

function Get-SafeName {
    param([string]$Path)
    $base = Split-Path -Leaf $Path
    if ([string]::IsNullOrWhiteSpace($base)) { $base = "images" }
    return ($base -replace "[^A-Za-z0-9._-]", "_")
}

function Resolve-ImageZone {
    param([string]$Dir)

    $name = (Split-Path -Leaf $Dir).ToLowerInvariant()
    $yMin = $ImageDefaultZoneYMin
    $yMax = $ImageDefaultZoneYMax

    if ($name -eq "image1") {
        $yMin = $Image1ZoneYMin
        $yMax = $Image1ZoneYMax
    } elseif ($name -eq "image5") {
        $yMin = $Image5ZoneYMin
        $yMax = $Image5ZoneYMax
    }

    if (-not [string]::IsNullOrWhiteSpace($ImageZoneYMinOverride)) { $yMin = $ImageZoneYMinOverride }
    if (-not [string]::IsNullOrWhiteSpace($ImageZoneYMaxOverride)) { $yMax = $ImageZoneYMaxOverride }

    return [pscustomobject]@{
        YMin = $yMin
        YMax = $yMax
    }
}

function Run-ImageCount {
    $dirs = Resolve-ImageDirs
    if (-not $dirs -or $dirs.Count -eq 0) {
        Write-Host "RUN_IMAGE_COUNT is enabled, but IMAGE_DIRS/BORDER_IMAGE_ROOT is not set. Skip image counting."
        return
    }

    New-Item -ItemType Directory -Force -Path $ImageCountRoot | Out-Null
    $idx = 0
    foreach ($dir in $dirs) {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) {
            throw "Image directory not found: $dir"
        }
        $idx += 1
        $label = Get-SafeName $dir
        $zone = Resolve-ImageZone $dir
        $outputCsv = Join-Path $ImageCountRoot "${idx}_${label}_vehicle_count_results_openvino.csv"
        $outputLog = Join-Path $ImageCountRoot "${idx}_${label}.log"

        Write-Host "Running image count ${idx}: $dir"
        Write-Host "Image count zone   : y=$($zone.YMin)..$($zone.YMax)"
        $args = @(
            (Join-Path $ScriptDir "count_vehicles_in_images_openvino.py"),
            "--image-dir", $dir,
            "--model", $ModelPath,
            "--conf", $ImageConf,
            "--output-csv", $outputCsv,
            "--reid-model-path", $ReidModelPath,
            "--openvino-device", $ImageOpenvinoDevice,
            "--zone-y-min", $zone.YMin,
            "--zone-y-max", $zone.YMax,
            "--reid-thresh", $ImageReidThresh,
            "--max-dist", $ImageMaxDist,
            "--lookback", $ImageLookback
        )
        if (-not [string]::IsNullOrWhiteSpace($ImageExtraArgs)) {
            $args += (Split-ConfiguredList $ImageExtraArgs)
        }
        Invoke-NativeLogged -LogPath $outputLog -FilePath $PythonBin -Arguments $args
        Write-Host "Image count CSV: $outputCsv"
        Write-Host "Image count log: $outputLog"
    }
}

New-Item -ItemType Directory -Force -Path $FullRoot, $PipelineLogDir, $RunRoot | Out-Null
Show-Config
Assert-RequiredFiles
Set-Location $RepoRoot

$gpuFiles = @()
if (Test-Enabled $RunDetection) {
    $gpuFiles = Run-DetectionTracking
    Merge-Manifests -GpuFiles $gpuFiles
} else {
    Write-Host "RUN_DETECTION is disabled. Reusing existing tracking root: $TrackingRoot"
    if (-not (Test-Path -LiteralPath $MergedManifest)) {
        throw "Merged manifest not found: $MergedManifest"
    }
}

if (Test-Enabled $RunGateCount) {
    Run-GateCount
} else {
    Write-Host "RUN_GATE_COUNT is disabled."
}

if (Test-Enabled $RunImageCount) {
    Run-ImageCount
} else {
    Write-Host "RUN_IMAGE_COUNT is disabled."
}

Write-Host "Full pipeline finished."
Write-Host "Tracking root    : $TrackingRoot"
Write-Host "Count root       : $CountRoot"
Write-Host "Manifest path    : $MergedManifest"
Write-Host "Image count root : $ImageCountRoot"
Write-Host "Worker logs      : $WorkerLogDir"

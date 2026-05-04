param(
    [Parameter(Mandatory = $true)]
    [string] $CudaVersion,

    [Parameter(Mandatory = $true)]
    [string] $ComponentsJson
)

$ErrorActionPreference = "Stop"

$platform = "windows-x86_64"
$baseUrl = "https://developer.download.nvidia.com/compute/cuda/redist"
$manifestUrl = "$baseUrl/redistrib_$CudaVersion.json"

$components = @(ConvertFrom-Json -InputObject $ComponentsJson)
if ($components.Count -eq 0) {
    throw "No CUDA redist components requested."
}

$toolCache = $env:RUNNER_TOOL_CACHE
if ([string]::IsNullOrWhiteSpace($toolCache)) {
    $toolCache = Join-Path $env:TEMP "runner-tool-cache"
}

$runnerTemp = $env:RUNNER_TEMP
if ([string]::IsNullOrWhiteSpace($runnerTemp)) {
    $runnerTemp = $env:TEMP
}

$cudaRoot = Join-Path $toolCache "cuda-redist\$CudaVersion\$platform"
$downloadRoot = Join-Path $toolCache "cuda-redist-downloads\$CudaVersion"
New-Item -ItemType Directory -Force -Path $cudaRoot, $downloadRoot | Out-Null

Write-Host "Reading CUDA redistrib manifest: $manifestUrl"
$manifest = Invoke-RestMethod -Uri $manifestUrl

foreach ($component in $components) {
    $marker = Join-Path $cudaRoot ".$component.complete"
    if (Test-Path -LiteralPath $marker) {
        Write-Host "CUDA component already extracted: $component"
        continue
    }

    $componentInfo = $manifest.$component
    if ($null -eq $componentInfo) {
        throw "CUDA redistrib manifest $CudaVersion does not contain component '$component'."
    }

    $packageInfo = $componentInfo.$platform
    if ($null -eq $packageInfo) {
        throw "CUDA component '$component' does not provide platform '$platform'."
    }

    $relativePath = $packageInfo.relative_path
    $url = "$baseUrl/$relativePath"
    $archive = Join-Path $downloadRoot (Split-Path -Leaf $relativePath)
    $expectedSha256 = $packageInfo.sha256.ToLowerInvariant()

    $needDownload = $true
    if (Test-Path -LiteralPath $archive) {
        $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
        $needDownload = $actualSha256 -ne $expectedSha256
    }

    if ($needDownload) {
        Write-Host "Downloading CUDA component: $component"
        Invoke-WebRequest -Uri $url -OutFile $archive
    } else {
        Write-Host "Using cached CUDA component archive: $component"
    }

    $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "Checksum mismatch for $archive. Expected $expectedSha256, got $actualSha256."
    }

    $extractRoot = Join-Path $runnerTemp "cuda-redist-$component"
    Remove-Item -LiteralPath $extractRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null

    Write-Host "Extracting CUDA component: $component"
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force

    $children = @(Get-ChildItem -LiteralPath $extractRoot)
    $sourceRoot = $extractRoot
    if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
        $sourceRoot = $children[0].FullName
    }

    Copy-Item -Path (Join-Path $sourceRoot "*") -Destination $cudaRoot -Recurse -Force
    Remove-Item -LiteralPath $extractRoot -Recurse -Force
    Remove-Item -LiteralPath $archive -Force
    New-Item -ItemType File -Force -Path $marker | Out-Null
}

$cudaBin = Join-Path $cudaRoot "bin"
$cudaLib = Join-Path $cudaRoot "lib\x64"

if (-not (Test-Path -LiteralPath (Join-Path $cudaRoot "include\cuda_runtime.h"))) {
    throw "CUDA redist root is missing include\cuda_runtime.h: $cudaRoot"
}

if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    "CUDA_PATH=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    "CUDA_HOME=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    "CUDAToolkit_ROOT=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}

if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_PATH)) {
    $cudaBin | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
    $cudaLib | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}

Write-Host "CUDA redist root: $cudaRoot"
if (Test-Path -LiteralPath (Join-Path $cudaBin "nvcc.exe")) {
    & (Join-Path $cudaBin "nvcc.exe") --version
}

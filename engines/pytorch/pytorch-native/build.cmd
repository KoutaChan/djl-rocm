@echo off
@rem Build djl_torch.dll, the JNI bridge between DJL's Java PtNDArrayEx
@rem layer and libtorch on Windows.
@rem
@rem Usage: build.cmd <VERSION> <FLAVOR>
@rem   VERSION  e.g. 2.11.0
@rem   FLAVOR   cpu / cu128 / cu130
@rem
@rem Prereqs (one-time, e.g. via chocolatey):
@rem   choco install cmake.install --installargs "ADD_CMAKE_TO_PATH=User" -y
@rem   choco install zulu21 -y

setlocal

set "VERSION=%~1"
set "FLAVOR=%~2"

@rem PT_VERSION is the cmake -D macro name (e.g. V1_13_X) for version-gated
@rem JNI branches, NOT the PyTorch version number. Clear any inherited env
@rem so a CI-exported PT_VERSION=<matrix.pt> cannot pollute the cmake flag
@rem with a non-identifier value like "2.11.0".
set "PT_VERSION="
set "USE_CUDA=0"

@rem
@rem libtorch download
@rem

set "DOWNLOAD_URL=https://download.pytorch.org/libtorch/%FLAVOR%/libtorch-win-shared-with-deps-%VERSION%%%2B%FLAVOR%.zip"
if exist libtorch (
    echo Found libtorch
) else (
    echo Downloading libtorch from: %DOWNLOAD_URL%
    powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; (New-Object Net.WebClient).DownloadFile('%DOWNLOAD_URL%', '%CD%\libtorch.zip')" || exit /b 1
    powershell -NoProfile -Command "Expand-Archive -LiteralPath libtorch.zip -DestinationPath '%CD%'" || exit /b 1
    del /f libtorch.zip
    echo Finished downloading libtorch
)

@rem
@rem PT_VERSION macro (only defined for the 1.13.x / 2.0.x / 2.1.x branches
@rem where JNI code gates on #ifdef V1_13_X)
@rem

for %%V in (1.13.1 2.0.1 2.1.1 2.1.2) do (
    if "%VERSION%" == "%%V" set "PT_VERSION=V1_13_X"
)

@rem
@rem USE_CUDA flag
@rem Note: batch substring (:~) only works on plain vars, not positional
@rem args, so we go through the FLAVOR local.
@rem

if /i "%FLAVOR:~0,2%" == "cu" set "USE_CUDA=1"

@rem
@rem Legacy libtorch patches (1.11 - 1.13 line only). The cuda.cmake
@rem override hard-codes an NVTX v1 nvToolsExt64_1.lib path that CUDA 12+
@rem no longer ships, and Parallel.h works around a VS 17.4 TLS link bug.
@rem PyTorch 2.x uses NVTX v3 header-only and needs neither, so only apply
@rem them when a version-specific Parallel.h is present.
@rem

if exist src\main\patch\%VERSION%\Parallel.h (
    copy /y src\main\patch\cuda.cmake libtorch\share\cmake\Caffe2\public\ || exit /b 1
    copy /y src\main\patch\%VERSION%\Parallel.h libtorch\include\ATen\ || exit /b 1
)

@rem
@rem Build
@rem

if exist build rd /q /s build
md build\classes
pushd build
javac -sourcepath ..\..\pytorch-engine\src\main\java\ ..\..\pytorch-engine\src\main\java\ai\djl\pytorch\jni\PyTorchLibrary.java -h include -d classes || goto :fail
cmake -DCMAKE_PREFIX_PATH=libtorch -DPT_VERSION="%PT_VERSION%" -DUSE_CUDA=%USE_CUDA% -DCMAKE_BUILD_TYPE=Release .. || goto :fail
cmake --build . --config Release || goto :fail

@rem MSBuild puts djl_torch.dll under build\Release\, but the gradle
@rem stageJniLib task looks for build\djl_torch.dll directly (same flat
@rem layout as the Makefile/Ninja generators on Linux/macOS). Surface it
@rem one level up so the packaging step picks it up.
if exist Release\djl_torch.dll copy /y Release\djl_torch.dll djl_torch.dll >nul

popd
endlocal
exit /b 0

:fail
popd
endlocal
exit /b 1

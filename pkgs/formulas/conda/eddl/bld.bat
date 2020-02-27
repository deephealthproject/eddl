:: CAREFUL!
:: THIS IS SKETCH, NOT TESTED!!!
:: IF YOU WANT TO COMPILE IT FOR WINDOWS, REVIEW ALL OF THIS CODE
@echo off

set TH_BINARY_BUILD=1

:: CUDA
if "%USE_CUDA%" == "0" (
    set build_with_cuda=
) else (
    set build_with_cuda=1
    set desired_cuda=%CUDA_VERSION:~0,-1%.%CUDA_VERSION:~-1,1%
)


if "%build_with_cuda%" == "" goto cuda_flags_end

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%desired_cuda%
set CUDA_BIN_PATH=%CUDA_PATH%\bin

:cuda_flags_end

:: MKL
curl https://s3.amazonaws.com/ossci-windows/mkl_2019.4.245.7z -k -O
7z x -aoa mkl_2019.4.245.7z -omkl
set CMAKE_INCLUDE_PATH=%SRC_DIR%\mkl\include
set LIB=%SRC_DIR%\mkl\lib;%LIB%

:: SCCACHE
IF "%USE_SCCACHE%" == "1" (
    mkdir %SRC_DIR%\tmp_bin
    curl -k https://s3.amazonaws.com/ossci-windows/sccache.exe --output %SRC_DIR%\tmp_bin\sccache.exe
    copy %SRC_DIR%\tmp_bin\sccache.exe %SRC_DIR%\tmp_bin\nvcc.exe
    set "PATH=%SRC_DIR%\tmp_bin;%PATH%"
    set SCCACHE_IDLE_TIMEOUT=1500
)

IF "%build_with_cuda%" == "" goto cuda_end

IF "%USE_SCCACHE%" == "1" (
    set CUDA_NVCC_EXECUTABLE=%SRC_DIR%\tmp_bin\nvcc
)

set "PATH=%CUDA_BIN_PATH%;%PATH%"

if "%CUDA_VERSION%" == "80" (
    :: Only if you use Ninja with CUDA 8
    set "CUDAHOSTCXX=%VS140COMNTOOLS%\..\..\VC\bin\amd64\cl.exe"
)

:cuda_end


IF NOT "%USE_SCCACHE%" == "1" goto sccache_end

sccache --stop-server
sccache --start-server
sccache --zero-stats

set CC=sccache cl
set CXX=sccache cl

:sccache_end


cmake -G "NMake Makefiles" -D BUILD_PROTOBUF=ON -D BUILD_EXAMPLES=OFF -D CMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% %SRC_DIR%
if errorlevel 1 exit 1

nmake
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1

IF "%USE_SCCACHE%" == "1" (
    taskkill /im sccache.exe /f /t || ver > nul
    taskkill /im nvcc.exe /f /t || ver > nul
)

if NOT "%build_with_cuda%" == "" (
    copy "%CUDA_BIN_PATH%\cudnn64_%CUDNN_VERSION%.dll*" %SP_DIR%\torch\lib
)

exit /b 0

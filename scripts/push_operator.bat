@echo off
REM Build and push operator image to ECR.
REM Run this manually for Week 6 before CI/CD is live.
REM Usage: scripts\push_operator.bat [tag]

setlocal

set REGION=eu-north-1
REM Account: ARGUS_ACCOUNT_ID if set, else whoever the AWS CLI is logged in as
if "%ARGUS_ACCOUNT_ID%"=="" for /f %%i in ('aws sts get-caller-identity --query Account --output text') do set ARGUS_ACCOUNT_ID=%%i
if "%ARGUS_ACCOUNT_ID%"=="" goto :error
set ACCOUNT=%ARGUS_ACCOUNT_ID%
set REPO=argus/operator
set TAG=%1
if "%TAG%"=="" set TAG=latest
set IMAGE=%ACCOUNT%.dkr.ecr.%REGION%.amazonaws.com/%REPO%:%TAG%

echo Logging into ECR...
aws ecr get-login-password --region %REGION% | docker login --username AWS --password-stdin %ACCOUNT%.dkr.ecr.%REGION%.amazonaws.com
if errorlevel 1 goto :error

echo Building operator image...
docker build -t "%IMAGE%" operator/
if errorlevel 1 goto :error

echo Pushing %IMAGE%...
docker push "%IMAGE%"
if errorlevel 1 goto :error

echo Done. Image: %IMAGE%
goto :end

:error
echo Error occurred during build/push!
exit /b 1

:end
endlocal

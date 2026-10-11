@echo off
setlocal
set "AGENT_PYTHON=%~dp0integration\.venv-agents-sdk\Scripts\python.exe"

if not exist "%AGENT_PYTHON%" (
    echo The project Python environment was not found. 1>&2
    echo See AGENT_OPERATIONS_AND_REGISTRY.md for the one-time setup instructions. 1>&2
    exit /b 1
)

"%AGENT_PYTHON%" "%~dp0agent_mvp\start_agent.py" %*
exit /b %errorlevel%

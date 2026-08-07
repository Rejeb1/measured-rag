@echo off
REM Convenience launcher.
REM
REM ragmed and its dependencies (torch+cu126, sentence-transformers) live in the
REM standalone Python 3.14 install, not in conda. An activated conda environment puts
REM its own python first on PATH, so `python -m ragmed.cli` inside `(base)` fails with
REM ModuleNotFoundError even though nothing is wrong with the project.
REM
REM This wrapper always calls the interpreter that actually has the package, so it
REM works from any shell and from any conda state.
REM
REM Usage:  ragmed ask "your question"
REM         ragmed serve
REM         ragmed eval --label mytest

setlocal

REM Allow an override, e.g. after moving to a venv:  set RAGMED_PYTHON=C:\path\python.exe
if defined RAGMED_PYTHON (
    set "PY=%RAGMED_PYTHON%"
) else (
    set "PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)

if not exist "%PY%" (
    echo [ragmed] Python interpreter not found: %PY%
    echo [ragmed] Set RAGMED_PYTHON to the interpreter that has ragmed installed.
    exit /b 1
)

"%PY%" -m ragmed.cli %*

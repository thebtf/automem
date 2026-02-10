@echo off
REM Start AutoMem Flask API in background, then launch MCP bridge
REM Used as MCP server command in claude_desktop_config.json

start /b "" "D:\Dev\forks\automem\.venv\Scripts\python.exe" "D:\Dev\forks\automem\app.py" >nul 2>&1

REM Wait for API to become ready
:wait_loop
timeout /t 1 /nobreak >nul
curl -s -o nul -w "" http://127.0.0.1:8001/health >nul 2>&1
if errorlevel 1 goto wait_loop

REM API is up — start MCP bridge (takes over stdio)
npx -y @verygoodplugins/mcp-automem

"""
Launcher Script: Starts FastAPI Backend & React Frontend
========================================================
Runs:
  - FastAPI server at http://127.0.0.1:8000
  - Vite React UI at http://localhost:5173
"""

import subprocess
import sys
import os
import time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")

def main():
    print("=" * 65)
    print("🚀 LAUNCHING NEURALFLIGHT AI (FastAPI + React)")
    print("=" * 65)

    # 1. Start FastAPI backend
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=BASE_DIR
    )

    # Give backend a moment to bind port
    time.sleep(2)

    # 2. Start React frontend
    print("[2/2] Starting React Vite Frontend on http://localhost:5173 ...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=UI_DIR,
        shell=True
    )

    time.sleep(2)
    print("\n" + "=" * 65)
    print("✔ Both services are running!")
    print("   • Backend API Docs : http://127.0.0.1:8000/docs")
    print("   • React Web App    : http://localhost:5173")
    print("=" * 65)
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        webbrowser.open("http://localhost:5173")
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()

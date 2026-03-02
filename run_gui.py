"""
Entry point for the Anti-Slop Regulator Platform Streamlit GUI.

Usage:
    python run_gui.py
    # or
    streamlit run gui/app.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "gui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "false"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

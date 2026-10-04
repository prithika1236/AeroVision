"""
Aircraft Lever Operator Ergonomics and Fatigue Monitoring System.
Note: The system has transitioned from Streamlit to a high-performance native Flask + HTML5/CSS3/Chart.js web dashboard.
Run `python main.py` to start the application.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print("[INFO] Launching Aircraft Lever Operator Monitoring System via main.py...")
    import main
    # Direct execution of main entry point
    main.app.run(
        host=main.config.SERVER_HOST,
        port=main.config.SERVER_PORT,
        debug=False,
    )

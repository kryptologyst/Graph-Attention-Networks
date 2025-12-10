#!/usr/bin/env python3
"""Launch script for the Graph Attention Networks demo."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "streamlit_demo.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        print("Please ensure the demo directory and streamlit_demo.py exist.")
        return 1
    
    print("Launching Graph Attention Networks Interactive Demo...")
    print("=" * 50)
    print("The demo will open in your web browser.")
    print("If it doesn't open automatically, navigate to the URL shown below.")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching demo: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Entry point for nanochess TUI.

Run with:
    python -m nanochess
    # or after pip install:
    nanochess
"""

import os
import sys
from pathlib import Path

from nanochess.tui.app import NanochessApp


def main():
    """Launch the nanochess TUI."""
    app = NanochessApp()
    app.run()

    # Check if a training command was queued
    if hasattr(app, '_training_command') and app._training_command:
        cmd = app._training_command

        # Clear the screen
        os.system('clear' if os.name != 'nt' else 'cls')

        # Print a nice header
        print("\n" + "="*70)
        print("  Starting Nanochess Training")
        print("="*70)
        print(f"\nCommand: {' '.join(cmd)}\n")

        # Change to project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Replace current process with the training command
        # This gives full terminal control (highlighting, ctrl+c, etc.)
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

"""Verify Python and dependency compatibility before training."""

import sys


def check_environment() -> bool:
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 10 or minor > 12:
        print(
            f"ERROR: Python {major}.{minor} detected. "
            "TensorFlow requires Python 3.10, 3.11, or 3.12.\n"
            "Install Python 3.12 from https://www.python.org/downloads/ "
            "and recreate the virtual environment:\n"
            '  py -3.12 -m venv venv\n'
            "  venv\\Scripts\\activate\n"
            "  pip install -r requirements.txt"
        )
        return False

    try:
        import tensorflow as tf  # noqa: F401
        import cv2  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

    print(f"Environment OK (Python {major}.{minor})")
    return True


if __name__ == "__main__":
    sys.exit(0 if check_environment() else 1)

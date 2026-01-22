"""Allow running the GUI as a module: python -m gui"""

import sys
from gui.app import main

if __name__ == "__main__":
    sys.exit(main())

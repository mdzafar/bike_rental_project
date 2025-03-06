import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from titanic_model.config.core import PACKAGE_ROOT, config
with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
    
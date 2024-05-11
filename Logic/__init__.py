import os
import sys
current_script_path = os.path.abspath(__file__)
indexer_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(indexer_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


# from Logic.core import *
# from Logic.utils import *

# from .core import *
# from .utils import *


__all__ = [k for k in globals().keys() if not k.startswith("_")]

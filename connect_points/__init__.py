import logging

from .active_contours import *
from .calculate_contours import *
from .graph_based import *
from .skeleton import *


# The code in this package will be removed after the complete migration to vt_tracker. It may not
# be up to date and stable anymore.
logging.warning("""
The package 'connect_points' will be replaced by vt_tracker, which can be installed from the
repository 'https://gitlab.inria.fr/vsouzari/vt_tracker.git'. This package might not be up to date
and stable anymore. Use it at your own risk.
""")

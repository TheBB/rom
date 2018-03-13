from nutils import config
from nutils.warnings import NutilsWarning
import multiprocessing

import warnings
warnings.filterwarnings('ignore', message='using explicit inflation; this is usually a bug', category=NutilsWarning)

# core.globalproperties['nprocs'] = multiprocessing.cpu_count()
# core.globalproperties['log'] = log._mklog()

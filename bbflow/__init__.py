from nutils import core
import multiprocessing

core.globalproperties['nprocs'] = multiprocessing.cpu_count()

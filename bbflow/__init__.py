from nutils import core, log
import multiprocessing

core.globalproperties['nprocs'] = multiprocessing.cpu_count()
core.globalproperties['log'] = log._mklog()

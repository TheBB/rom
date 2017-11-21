import inspect
import functools
import time as timemod
import numpy as np
import scipy as sp
from nutils import log


def multiple_to_single(argname):
    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def ret(*args, **kwargs):
            binding = signature.bind(*args, **kwargs)
            args = binding.arguments[argname]
            multiple = True
            if not isinstance(args, (list, tuple)):
                multiple = False
                args = [args]
            retvals = []
            for arg in args:
                binding.arguments[argname] = arg
                retvals.append(func(*binding.args, **binding.kwargs))
            if not multiple:
                return retvals[0]
            return retvals

        return ret
    return decorator


class time:

    def __init__(self, context=None, display=True):
        self._display = True
        self._context = context
        self._time = 0.0

    def __enter__(self):
        self._time = timemod.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._time = timemod.time() - self._time
        if self._display:
            s = (str(self._context) + ': ') if self._context is not None else ''
            log.user('{}{:.2e} seconds'.format(s, self._time))

    @property
    def seconds(self):
        return self._time


def collocate(domain, equation, points, index, size):
    ncomps = equation.shape[-1]

    elements = [domain.elements[eid] for eid, __ in points]
    kwargs = [{
        '_transforms': (elem.transform, elem.opposite),
        '_points': np.array([pt]),
    } for elem, (__, pt) in zip(elements, points)]

    data = np.array([equation.eval(**kwg)[0] for kwg in kwargs])

    if equation.ndim == 2:
        data = np.transpose(data, (0, 2, 1))
        data = np.reshape(data, (ncomps * len(points), data.shape[-1]))
        data = sp.sparse.coo_matrix(data)
        data = sp.sparse.csr_matrix((data.data, (data.row + index, data.col)), shape=(size,)*2)
    elif equation.ndim == 1:
        data = np.hstack([np.zeros((index,)), data.flatten()])
    else:
        raise NotImplementedError

    return data

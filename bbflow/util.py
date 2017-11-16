import inspect
import functools
import time as timemod


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

    def __init__(self, display=True):
        self._display = True
        self._time = 0.0

    def __enter__(self):
        self._time = timemod.time()

    def __exit__(self):
        self._time = timemod.time() - self._time

    @property
    def seconds(self):
        return self._time

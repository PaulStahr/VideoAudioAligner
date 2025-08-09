from functools import lru_cache, wraps
from threading import Lock
from inspect import isfunction

class FunctionWrapper:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._lock = Lock()

    def __call__(self):
        with self._lock:
            if self._result is None:
                self._result = self.func(*self.args, **self.kwargs)
                self.__call__ = lambda: self._result
        return self._result

class IgnoredArg:
    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        return isinstance(other, IgnoredArg) and self.tag == other.tag

def threadsafe_lru_cache(func=None, *, maxsize=128):
    def decorator(f):
        @lru_cache(maxsize=maxsize)
        def make_wrapper(*args, **kwargs):
            return FunctionWrapper(f, *args, **kwargs)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return make_wrapper(*args, **kwargs)()
        return wrapper

    if func is not None and isfunction(func):
        # Called directly: threadsafe_lru_cache(func)
        return decorator(func)
    else:
        # Called with params: @threadsafe_lru_cache(maxsize=...)
        return decorator

def test_threadsafe_lru_cache():
    import time
    x = [0]
    @threadsafe_lru_cache()
    def slow_func(i):
        time.sleep(1)
        if x[0] != 0:
            raise Exception("Function was called more than once")
        x[0] += i
        return x[0]

    #Call function simoultaneously from multiple threads
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(slow_func, 5) for i in range(3)]
    #Wait for all threads to finish
    for future in futures:
        future.result()
    # Check that the function was only called once
    assert x[0] == 5
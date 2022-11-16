from collections.abc import Callable
from typing import Any

def func_time(f: Callable, *args: Any) -> float:
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic
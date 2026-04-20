import time

class Timer:
    def __init__(self, name="block"):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[TIMER] {self.name}: {elapsed:.6f} s")

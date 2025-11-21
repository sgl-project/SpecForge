class BenchmarkRegistry:

    def __init__(self):
        self.benchmarks = {}

    def register(self, name: str):
        """
        Usage:
        ```python
            BENCHMARKS = BenchmarkRegistry()

            BENCHMARKS.register("aime")
            class AIMEBenchmarker(Benchmarker):
                ...
        ```
        """

        def wrapper(cls):
            self.benchmarks[name] = cls
            return cls

        return wrapper

    def get(self, name: str) -> type:
        """
        Get the benchmark class by name.
        """
        return self.benchmarks[name]


BENCHMARKS = BenchmarkRegistry()

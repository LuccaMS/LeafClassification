from time import perf_counter


def count_time(fn):
    def wrapper(*args, **kwargs):
        before = perf_counter()

        response = fn(*args, **kwargs)
        print(perf_counter() - before)
        return response

    return wrapper

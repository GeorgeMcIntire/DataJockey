import itertools


def chunk_dict(d: dict, size: int):
    """
    Yield successive sub-dictionaries of given size.
    """
    it = iter(d.items())
    for i in range(0, len(d), size):
        yield dict(list(itertools.islice(it, size)))

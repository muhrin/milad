import numpy

__all__ = 'generate_all_pair_distances'


def calculate_all_pair_distances(vectors, sort_result=True):
    """Calculate all pair distances between the given vectors"""
    num = len(vectors)
    lengths = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            dr = vectors[i] - vectors[j]
            lengths.append(numpy.linalg.norm(dr))

    if sort_result:
        lengths.sort()
    return lengths

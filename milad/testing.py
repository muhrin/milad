import numpy

def generate_vectors_on_sphere(num):
    """Generate `num` random unit vectors on a sphere"""
    vecs = numpy.empty((num, 3))
    for i in range(num):
        vec = numpy.random.rand(3)
        vec /= numpy.linalg.norm(vec, axis=0)
        vecs[i] = vec
    return vecs
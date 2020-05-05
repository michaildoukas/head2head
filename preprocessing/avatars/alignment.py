# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved

import numpy
from . import mathutils


def compute_alignment_point_sets_3D(source, target):
    """Compute transform (Scale Rotation Translation) which aligns the source point set to the target point set

    Reference: https://github.com/oleg-alexandrov/projects/blob/master/eigen/Kabsch.cpp

    source -- 3xN numpy array with source points in 3D space
    target -- 3xN numpy with with target points in 3D space following the same pattern as source

    Returns (Rotation Translation Scale) tuple where
    - Rotation is a 3x3 numpy array with the rotation matrix
    - Translation is a 3x1 numpy array with the translation vector
    - Scale is a float with the uniform scale
    """

    if not isinstance(source, numpy.ndarray):
        raise TypeError("source is not a numpy-ndarray")
    if not isinstance(target, numpy.ndarray):
        raise TypeError("target is not a numpy-ndarray")
    if source.shape[0] != 3 or len(source.shape) != 2:
        raise ValueError("source is not an array of 3D points")
    if target.shape[0] != 3 or len(target.shape) != 2:
        raise ValueError("target is not an array of 3D points")
    if target.shape[1] != source.shape[1]:
        raise ValueError("source and target do not have the same number of points")
    
    num_points = source.shape[1]

    # compute centroids
    src_centroid = mathutils.sum_point_set(source) / num_points
    tgt_centroid = mathutils.sum_point_set(target) / num_points

    src_centered = mathutils.subtract_vector3_from_point_set(source, src_centroid)
    tgt_centered = mathutils.subtract_vector3_from_point_set(target, tgt_centroid)

    # estimate scale
    scale = numpy.linalg.norm(tgt_centered) / numpy.linalg.norm(src_centered)
    
    # scale the source to match the target before computing rotation
    src_centered = scale * src_centered
    src_centroid = scale * src_centroid

    # compute optimal rotation from the SVD of the correlation matrix
    correlation = numpy.matmul(src_centered, tgt_centered.transpose())
    U, _D, Vt = numpy.linalg.svd(correlation)
    V = Vt.transpose()
    # handle possible reflection
    d = numpy.sign(numpy.linalg.det(numpy.matmul(V, U.transpose())))
    E = numpy.identity(3)
    E[2, 2] = d
    R = numpy.matmul(V, numpy.matmul(E, U.transpose()))

    rotation = R
    T = tgt_centroid - numpy.matmul(R, src_centroid)

    return rotation, T, scale




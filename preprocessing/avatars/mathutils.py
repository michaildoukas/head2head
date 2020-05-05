# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved

import numpy


# Utility module with functions dealing with math operations on points sets, vector3, transformation matrices, etc
# Functions here follow consistent conventions and should be prefered 

# TODO: define point set class?

def subtract_vector3_from_point_set(points, vec3):
    '''Subtract a vec3 from all the points in the input set

    points -- 3xN numpy array with source points in 3D space
    '''
    return (points.transpose() - vec3).transpose()

def add_vector3_to_point_set(points, vec3):
    '''Add a vec3 from all the points in the input set

    points -- 3xN numpy array with source points in 3D space
    '''
    return (points.transpose() + vec3).transpose()

def sum_point_set(points):
    '''Sum all the points in the input set

    points -- 3xN numpy array with source points in 3D space
    '''
    return numpy.sum(points, axis=1)

def apply_transform3x3_to_point_set(transform, points):
    return numpy.dot(transform, points)

def apply_affine_transform_to_point_set(R, T, S, points):
    temp = apply_transform3x3_to_point_set(numpy.matmul(numpy.diag(S), R), points)
    return add_vector3_to_point_set(temp, T)


def apply_transform3x3_to_vector3(transform, vec3):
    return numpy.dot(transform, vec3)

def apply_affine_transform_to_vector3(R, T, S, vec3):
    return apply_transform3x3_to_vector3(numpy.matmul(numpy.diag(S), R), vec3) + T
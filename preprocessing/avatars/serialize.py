# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved

import pickle
import json
import numpy        # needs to be installed
import flatbuffers  # needs to be installed
from flatbuffers.number_types import UOffsetTFlags

# types generated from schema 
from .FacesoftFlatbuffersSchema import SerializedTriMesh
from .FacesoftFlatbuffersSchema import SerializedMorphableModel
from .FacesoftFlatbuffersSchema import SerializedComponent



def get_version():
    """Returns single integer number with the serialization version"""
    return 2

def deserialize_binary_to_morphable_model(data_blob):
    """De-serialize a binary blob as defined by flatbuffers

    data_blob -- Binary blob created by @serialize_morphable_model_to_binary
    Returns a dict with key-value pairs as described for the input arguments in serialize_morphable_model_to_binary
    """

    buf = bytearray(data_blob)
    mmodel = SerializedMorphableModel.SerializedMorphableModel.GetRootAsSerializedMorphableModel(buf, 0)

    mean_points = mmodel.MeanMesh().PointsAsNumpy()
    mean_indices = mmodel.MeanMesh().TriIndicesAsNumpy()
    weights = []
    components = []
    for i in range(mmodel.ComponentsLength()):
        c = mmodel.Components(i)
        weights.append(c.Scale())
        components.append(c.PointsAsNumpy())

    if len(components)> 0:
        components = numpy.asarray(components, dtype=numpy.float32)
    if len(weights) > 0:
        weights = numpy.asarray(weights, dtype=numpy.float32).ravel()

    submesh_offsets = []
    if mmodel.MeanMesh().SubmeshIndexOffsetLength() > 0:
        submesh_offsets = mmodel.MeanMesh().SubmeshIndexOffsetAsNumpy()

    dict = {'mean_points':mean_points, 'mean_indices':mean_indices, 'weights':weights, 'components':components, 'submesh_offsets':submesh_offsets}
    if mmodel.MeanMesh().UVLength() > 0:
        dict['mean_uvs'] = mmodel.MeanMesh().UVAsNumpy()

    return dict


def serialize_morphable_model_to_binary(mean_points, mean_indices, components, weights, mean_uvs=None, submesh_offsets=None):
    """Serialize the Morphable Model defined by the input vectors to the Flatbuffers Schema used in the package

    mean_points -- 1xN numpy array of floats with the 3D coordinates of the mean mesh points where N = 3 * num_points
    mean_indices -- (optional) flat numpy array of ints with the indices of the mean mesh triangles 
    referencing the mean_points 
    mean_uvs -- (optional) flat numpy array of floats with the UV coords for each mean mesh vertex 
    components -- MxN numpy array of floats with the 3D coordinates of each component of the Morphable Model where M is 
    the total number of components
    weights -- 1xM numpy array of floats with the weight factor for each component  

    Note that the first dimension of components need to be the same as the weights while the 
    second dimension needs to be the same as the mean_points length.
    Returns the serialized binary blob. 
    """

    if mean_points is not None:
        if not isinstance(mean_points, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if mean_indices is not None:
        if not isinstance(mean_indices, numpy.ndarray):
           raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if mean_uvs is not None:
        if not isinstance(mean_uvs, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if submesh_offsets is not None:
        if not isinstance(submesh_offsets, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if components is not None:
        if not isinstance(components, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if weights is not None:
        if not isinstance(weights, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")

    if components is not None:
        if weights is not None:
            if len(weights) != components.shape[0]:
                raise ValueError("number of weight values should match total components")
        if mean_points is not None:
            if mean_points.size != components.shape[1]:
                raise ValueError("number of mean points should match component points")

    builder = flatbuffers.Builder(1024)

    # add each component to the builder first
    if components is not None:
        N_components = components.shape[0]
    else:
        N_components = 0
    b_components = []
    if components is not None:
        for i in reversed(range(N_components)):
            if weights is not None:
                weight = weights[i]
            else:
                weight = 1.

            Uvector = components[i]

            x = builder.CreateNumpyVector(Uvector)
            SerializedComponent.SerializedComponentStart(builder)
            SerializedComponent.SerializedComponentAddPoints(builder, x)
            SerializedComponent.SerializedComponentAddScale(builder, weight)
            b_components.append(SerializedComponent.SerializedComponentEnd(builder))

    SerializedMorphableModel.SerializedMorphableModelStartComponentsVector(builder, N_components)
    for c in b_components:
        builder.PrependUOffsetTRelative(c)
    all_components = builder.EndVector(N_components)

    # add the raw data of the mean mesh first
    if mean_points is not None:
        builder_points = builder.CreateNumpyVector(mean_points)
    else:
        builder_points = builder.CreateNumpyVector(numpy.array([], dtype=numpy.float32))
    if mean_indices is not None:
        builder_tri_indices = builder.CreateNumpyVector(mean_indices)
    else:
        builder_tri_indices = builder.CreateNumpyVector(numpy.array([], dtype=numpy.float32))
    if mean_uvs is not None:
        builder_uvs = builder.CreateNumpyVector(mean_uvs)
    if submesh_offsets is not None:
        builder_submeshes = builder.CreateNumpyVector(submesh_offsets)

    # add the mean mesh
    SerializedTriMesh.SerializedTriMeshStart(builder)
    #if mean_points is not None:
    SerializedTriMesh.SerializedTriMeshAddPoints(builder, builder_points)
    #if mean_indices is not None:
    SerializedTriMesh.SerializedTriMeshAddTriIndices(builder, builder_tri_indices)
    if mean_uvs is not None:
        SerializedTriMesh.SerializedTriMeshAddUV(builder, builder_uvs)
    if submesh_offsets is not None:
        SerializedTriMesh.SerializedTriMeshAddSubmeshIndexOffset(builder, builder_submeshes)
    builder_meanMesh = SerializedTriMesh.SerializedTriMeshEnd(builder)

    # aggregate everything to the MModel
    SerializedMorphableModel.SerializedMorphableModelStart(builder)
    SerializedMorphableModel.SerializedMorphableModelAddVersion(builder, get_version())
    SerializedMorphableModel.SerializedMorphableModelAddMeanMesh(builder, builder_meanMesh)
    SerializedMorphableModel.SerializedMorphableModelAddComponents(builder, all_components)
    builder_mmodel = SerializedMorphableModel.SerializedMorphableModelEnd(builder)

    builder.Finish(builder_mmodel)

    return builder.Output()

def serialize_from_dictionary(data, uv_data=[]):
    """ Helper function to serialize the input dict data into the package flatbuffers format
    All keys are optional but at least there should be at least valid points OR components

    data -- expected input must be a dict with the following keys
    - points (optional) -- 3D vertices of the mean mesh (will get flattened)
    - trilist (optional) -- index triplets with the triangles in the main sub mesh of the mean mesh (will get flattened)
    - std (optional) -- standard deviation for each component (will get the sqrt of these values)
    - components (optional) -- components of the model as NumComponents x NumPoints
    - submeshes (optional) -- extract lists of triangle indices with further submeshes in the mean mesh (will get flattened)
    - uv_data -- dict with tcoords key with all UVs for the mean points (will get flattened)

    Return the serialized blob that can be stored into disk.
    """

    if not isinstance(data, dict):
        raise TypeError("non dict data passed to serialize_from_dictionary")
    if uv_data:
        if not isinstance(uv_data, dict):
            raise TypeError("non dict uv data passed to serialize_from_dictionary")

    if 'points' in data.keys():
        points = numpy.asarray(data['points'], dtype=numpy.float32).ravel()
    else:
        points = None
    
    if 'trilist' in data.keys():
        # this needs to be uint32 as uint16 doesn't seem to be natively supported by flatbuffers
        tri_indices = numpy.asarray(data['trilist'], dtype=numpy.uint32).ravel()
    else:
        tri_indices = None

    if 'submeshes' in data.keys() and tri_indices is not None:
        submesh_offsets = numpy.zeros(len(data['submeshes']), dtype=numpy.uint32)
        for (i, submesh) in enumerate(data['submeshes']):
            submesh_indices = numpy.asarray(submesh, dtype=numpy.uint32)
            if submesh_indices.size % 3 != 0:
                raise TypeError("invalid sub mesh size")
            # append the sub mesh indices            
            offset = tri_indices.size
            submesh_offsets[i] = offset
            tri_indices = numpy.append(tri_indices, submesh_indices.ravel())
    else:
        submesh_offsets = None

    # add std if in data
    if 'std' in data.keys():
        std = numpy.asarray(data['std'], dtype=numpy.float32)
        # need to take the square root of the std as scale factor
        std = numpy.sqrt(std)
    else:
        std = None
    
        # add components if needed
    if 'components' in data.keys():
        components = numpy.asarray(data['components'], dtype=numpy.float32)
    else:
        components = None

    uvs = None
    if uv_data:
        uvs = numpy.asarray(uv_data["tcoords"], dtype=numpy.float32).ravel()

    # check for missing data
    if components is None and points is None:
        raise TypeError("dictionary passed to serialize_from_dictionary does not contain either points nor components")

    # serialize into flatbuffers
    dataBlob = serialize_morphable_model_to_binary(points, tri_indices, components, std, uvs, submesh_offsets)

    return dataBlob

def serialize_from_dictionary_file(inFile, outFile, uvFile=[]):
    """Helper to serialize Morphable Model from a dictionary saved in a pickle file and write the result into the output file"""

    with open(inFile, 'rb') as file:
        d = pickle.load(file)

    uv_dict = []
    if uvFile:
        with open(uvFile) as f:
            uv_dict = json.load(f)

    dataBlob = serialize_from_dictionary(d, uv_dict)

    with open(outFile,'wb') as f:
        f.write(dataBlob)


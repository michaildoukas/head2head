# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved

import numpy

import flatbuffers


def get_version():
    """Returns version of timeline serialize"""
    return 1

def serialize_frames_to_binary(frames, dt=1/30):
    """ Serialize key-frames stacked horizontally over time 
    
    """

    if frames is not None:
        if not isinstance(frames, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to serialize_frames")
    
    if (dt < 0.):
        raise TypeError("time period dt should be positive real number")
    
    from .FacesoftFlatbuffersSchema import SerializedKeyframesTimeline

    # samples per frame
    N = frames.shape[1]

    builder = flatbuffers.Builder(1024)
    f = builder.CreateNumpyVector(frames.ravel().astype(numpy.float32))

    SerializedKeyframesTimeline.SerializedKeyframesTimelineStart(builder)
    SerializedKeyframesTimeline.SerializedKeyframesTimelineAddKeyframesData(builder, f)
    SerializedKeyframesTimeline.SerializedKeyframesTimelineAddVersion(builder, get_version())
    SerializedKeyframesTimeline.SerializedKeyframesTimelineAddTimeBetweenFramesSecs(builder, dt)
    SerializedKeyframesTimeline.SerializedKeyframesTimelineAddSamplesPerFrame(builder, N)


    serializedData = SerializedKeyframesTimeline.SerializedKeyframesTimelineEnd(builder)

    builder.Finish(serializedData)

    return builder.Output()

def deserialize_binary_to_frames(data_blob):
    """De-serialize the input binary data to a 2D array of keyframes"""

    from .FacesoftFlatbuffersSchema import SerializedKeyframesTimeline

    buf = bytearray(data_blob)
    serialized_keyframes = SerializedKeyframesTimeline.SerializedKeyframesTimeline.GetRootAsSerializedKeyframesTimeline(buf, 0)

    dt = serialized_keyframes.TimeBetweenFramesSecs()
    dim2 = serialized_keyframes.SamplesPerFrame()
    # dim1 = serialized_keyframes.KeyframesDataLength() / dim2

    keyframes = serialized_keyframes.KeyframesDataAsNumpy()
    keyframes = keyframes.reshape((-1, dim2))

    return keyframes, dt

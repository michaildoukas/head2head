# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved

"""Facesoft avatars package
For internal use only.

Python package with modules implementing utilities for the Avatars native module.
See https://github.com/Facesoft-github/facesoft-main-repo-native/tree/master/dev/avatars.


avatars_bindings
    The avatars_bindings is imported (if available) as part of the avatars package.
    It contains wrappers to native implementations, mostly for mesh processing.

"""

# Try to import bindings
try:
    from . import avatars_bindings
except:
    import warnings
    warnings.warn("Failed to import avatars_bindings module", ImportWarning)
    mesh = None
import os
import json
import sys
import importlib

# Default backend: eddl.
_BACKEND = 'eddl'

if _BACKEND == "eddl":
    sys.stderr.write('Using Eddl backend.\n')
    from .eddl_backend import *

else:
    # Try and load external backend.
    try:
        backend_module = importlib.import_module(_BACKEND)
        entries = backend_module.__dict__
        # Check if valid backend.
        # Module is a valid backend if it has the required entries.
        required_entries = ['placeholder', 'variable', 'function']
        for e in required_entries:
            if e not in entries:
                raise ValueError('Invalid backend. Missing required entry : ' + e)
        namespace = globals()
        for k, v in entries.items():
            # Make sure we don't override any entries from common, such as epsilon.
            if k not in namespace:
                namespace[k] = v
        sys.stderr.write('Using ' + _BACKEND + ' backend.\n')
    except ImportError:
        raise ValueError('Unable to import backend : ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend PyEddl is currently using.
    # Example
    ```python
        >>> pyeddl.backend.backend()
        'eddl'
    ```
    """
    return _BACKEND

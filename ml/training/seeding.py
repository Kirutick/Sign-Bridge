import os
import random
import numpy as np
import tensorflow as tf

def set_deterministic_seeds(seed: int = 42):
    """
    Pins seeds for python random, numpy, and tensorflow.
    Configures TensorFlow for deterministic ops where possible.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Attempt to force deterministic GPU operations
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # Older TF versions might not support this
        pass
        
    return seed

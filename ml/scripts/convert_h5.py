import sys
import numpy as np

np.object = np.object_
np.bool = np.bool_

class MockHub:
    pass
sys.modules['tensorflow_hub'] = MockHub()

import tensorflow as tf
if not hasattr(tf.compat.v1, 'estimator'):
    tf.compat.v1.estimator = type('estimator', (), {'Exporter': object})

from tensorflowjs.converters.converter import convert
convert(['--input_format=keras', 'models/sign_bridge_v2/temp.h5', 'models/sign_bridge_v2/model_layers'])

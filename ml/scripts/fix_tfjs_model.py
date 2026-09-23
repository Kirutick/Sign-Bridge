import json
import sys

model_path = sys.argv[1]
with open(model_path, 'r') as f:
    data = json.load(f)

for layer in data['modelTopology']['model_config']['config']['layers']:
    if layer['class_name'] == 'InputLayer':
        if 'batch_shape' in layer['config'] and 'batch_input_shape' not in layer['config']:
            layer['config']['batch_input_shape'] = layer['config']['batch_shape']
            
with open(model_path, 'w') as f:
    json.dump(data, f)

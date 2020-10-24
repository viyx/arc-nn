# import json
# import os
# import numpy as np


# def read_arc():
#     eval_folder = './data/evaluation'
#     train_folder = './data/training'

#     for t in os.listdir(train_folder):
#         with open(os.path.join(train_folder, t)) as f:
#             j = json.load(f)
#             train_inputs = np.array([t['input'] for t in j['train']])
#             train_outputs = np.array([t['output'] for t in j['train']])
#             test_inputs = np.array([t['input'] for t in j['test']])
#             test_output = np.array([t['output'] for t in j['test']])
#     return train_inputs, train_outputs




import numpy as np
anchor1 = np.array([44.8116, 221.419, 51.9056, 262.578, 63.0782, 319.037]).reshape((-1, 2))
anchor2 = np.array([21.6847, 104.836, 25.4569, 138.114, 37.0575, 180.267]).reshape((-1, 2))
anchor3 = np.array([8.77214, 37.8896, 12.134,  58.4511, 15.4512, 82.0343]).reshape((-1, 2))
anchors = [anchor1, anchor2, anchor3]
nx = [16, 32, 64]
ny = [16, 32, 64]
strides = [32,16,8]
class_nums = 1
arch = 34
paras = {'nx': nx, 'ny': nx, 'anchors': anchors, 'nc': class_nums, 'strides': strides}
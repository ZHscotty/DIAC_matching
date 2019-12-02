import pandas as pd
import numpy as np
import jieba
import tensorflow as tf

inputs = np.random.random(size=(63, 63, 16))
inputs_pad = tf.pad(inputs, paddings=[[3, 3], [3, 3], [0, 0]])
print(inputs_pad.shape)
conv = tf.layers.conv2d(inputs_pad, filters=32, kernel_size=(7, 7), strides=(1, 1))
print(conv.shape)



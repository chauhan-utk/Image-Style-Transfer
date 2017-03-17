import tensorflow as tf
import numpy as np

def net(input_image):
    conv1 = conv_layer(input_image, 9, 32, 1)
    conv1_relu = relu_layer(conv1)
    
    conv2 = conv_layer(conv1_relu, 3, 64, 2)
    conv2_relu = relu_layer(conv2)
    
    conv3 = conv_layer(conv2_relu, 3, 128, 2)
    conv3_relu = relu_layer(conv3)
    
    resid1 = resid_layer(conv3_relu, 3, 128, 1)
    resid2 = resid_layer(conv3_relu, 3, 128, 1)
    resid3 = resid_layer(conv3_relu, 3, 128, 1)
    resid4 = resid_layer(conv3_relu, 3, 128, 1)
    resid5 = resid_layer(conv3_relu, 3, 128, 1)
    
    conv_trans1 = conv_trans_layer(resid5, 3, 64, 2)
    conv_trans1_relu = relu_layer(conv_trans1)
    conv_trans2 = conv_trans_layer(conv_trans1_relu, 3, 32, 2)
    conv_trans2_relu = relu_layer(conv_trans2)
    conv_trans3 = conv_layer(conv_trans2_relu, 9, 3, 1)
    
    out = 255.0 / 2 * (tf.nn.tanh(conv_trans3) + 1)
    #out = tf.nn.tanh(conv_trans3) * 150 + 255.0/2
    #print(input_image)
    #print(out)
    return out
    
    
    
    
def conv_layer(input, filter_size, out_channels, stride):
    _, height, width, in_channels = [i.value for i in input.get_shape()]
    filter_shape = (filter_size, filter_size, in_channels, out_channels)
    filter = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1, seed=1), dtype=tf.float32)
    strides = [1, stride, stride, 1]
    conv = tf.nn.conv2d(input, filter, strides, 'SAME')
    return instance_norm(conv)
    
def conv_trans_layer(input, filter_size, out_channels, stride):
    batch_size, height, width, in_channels = [i.value for i in input.get_shape()]
    filter_shape = (filter_size, filter_size, out_channels, in_channels)
    filter = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1, seed=1), dtype=tf.float32)
    new_shape = tf.pack([batch_size, height * stride, width * stride, out_channels])
    strides = [1, stride, stride, 1]
    conv = tf.nn.conv2d_transpose(input, filter, new_shape, strides, 'SAME')
    return instance_norm(conv)
    
def resid_layer(input, filter_size, out_channels, stride):
    conv1 = conv_layer(input, filter_size, out_channels, stride)
    conv1_relu = relu_layer(conv1)
    conv2 = conv_layer(conv1_relu, filter_size, out_channels, stride)
    return input + conv2

def relu_layer(input):
    return tf.nn.relu(input)

def instance_norm(input):
    batch_size, height, width, in_channels = [i.value for i in input.get_shape()]
    mu, sigma = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros([in_channels]))
    scale = tf.Variable(tf.ones([in_channels]))
    epsilon = 1e-9
    normalized = (input - mu) / (sigma + epsilon) ** (0.5)
    return scale * normalized + shift


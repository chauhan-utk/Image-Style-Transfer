import tensorflow as tf
import numpy as np
import scipy.io

# Defines the architecture of VGG-19 neural network as defined in the paper
# Return the network model and the mean pixel
def net(data_path, input_image):

    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    net = {}
    current = input_image

    
    #### CONV1_1
    kernels, bias = weights[0][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv1_1'] = current
    #### RELU1_1
    current = tf.nn.relu(current)
    net['relu1_1'] = current
    #### CONV1_2
    kernels, bias = weights[2][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv1_2'] = current
    #### RELU1_2
    current = tf.nn.relu(current)
    net['relu1_2'] = current
    #### POOL1
    current = pool_layer(current)
    net['pool1'] = current
    
    
    
    #### CONV2_1
    kernels, bias = weights[5][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv2_1'] = current
    #### RELU2_1
    current = tf.nn.relu(current)
    net['relu2_1'] = current
    #### CONV2_2
    kernels, bias = weights[7][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv2_2'] = current
    #### RELU2_2
    current = tf.nn.relu(current)
    net['relu2_2'] = current
    #### POOL2
    current = pool_layer(current)
    net['pool2'] = current
    
    
    
    #### CONV3_1
    kernels, bias = weights[10][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv3_1'] = current
    #### RELU3_1
    current = tf.nn.relu(current)
    net['relu3_1'] = current
    #### CONV3_2
    kernels, bias = weights[12][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv3_2'] = current
    #### RELU3_2
    current = tf.nn.relu(current)
    net['relu3_2'] = current
    #### CONV3_3
    kernels, bias = weights[14][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv3_3'] = current
    #### RELU3_3
    current = tf.nn.relu(current)
    net['relu3_3'] = current
    #### CONV3_4
    kernels, bias = weights[16][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv3_4'] = current
    #### RELU3_4
    current = tf.nn.relu(current)
    net['relu3_4'] = current
    #### POOL3
    current = pool_layer(current)
    net['pool3'] = current
    
    
        
    #### CONV4_1
    kernels, bias = weights[19][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv4_1'] = current
    #### RELU4_1
    current = tf.nn.relu(current)
    net['relu4_1'] = current
    #### CONV4_2
    kernels, bias = weights[21][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv4_2'] = current
    #### RELU4_2
    current = tf.nn.relu(current)
    net['relu4_2'] = current
    #### CONV4_3
    kernels, bias = weights[23][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv4_3'] = current
    #### RELU4_3
    current = tf.nn.relu(current)
    net['relu4_3'] = current
    #### CONV4_4
    kernels, bias = weights[25][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv4_4'] = current
    #### RELU4_4
    current = tf.nn.relu(current)
    net['relu4_4'] = current
    #### POOL4
    current = pool_layer(current)
    net['pool4'] = current
    
    
    
    #### CONV5_1
    kernels, bias = weights[28][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv5_1'] = current
    #### RELU5_1
    current = tf.nn.relu(current)
    net['relu5_1'] = current
    #### CONV5_2
    kernels, bias = weights[30][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv5_2'] = current
    #### RELU5_2
    current = tf.nn.relu(current)
    net['relu5_2'] = current
    #### CONV5_3
    kernels, bias = weights[32][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv5_3'] = current
    #### RELU5_3
    current = tf.nn.relu(current)
    net['relu5_3'] = current
    #### CONV5_4
    kernels, bias = weights[34][0][0][0][0]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = conv_layer(current, kernels, bias)
    net['conv5_4'] = current
    #### RELU5_4
    current = tf.nn.relu(current)
    net['relu5_4'] = current
    
    return net, mean_pixel


    
    

def conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def pool_layer(input):
    return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')


def subtract_mean(image, mean_pixel):
    return image - mean_pixel


def add_mean(image, mean_pixel):
    return image + mean_pixel
    
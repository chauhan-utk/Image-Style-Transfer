import vgg
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LOSS_WEIGHT = 1e-3
STYLE_LOSS_WEIGHT = 1e0
LEARNING_RATE = 1e1

def stylize(network_path, content, style, iterations):

    # Append an extra batch number dimension which is required by tensorflow
    content_shape = (1,) + content.shape	
    style_shape = (1,) + style.shape
    
    # Dictionaries to hold values for content and style features
    content_features = {}
    style_features = {}

    
    # Compute content and style features using forward passes
    with tf.Graph().as_default(), tf.Session() as sess:
        print('Start getting content features')
        image = tf.placeholder('float', shape = content_shape)
        net, mean_pixel = vgg.net(network_path, image)
        content_pre = np.array([vgg.subtract_mean(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    with tf.Graph().as_default(), tf.Session() as sess:
        print('Start getting style features')
        image = tf.placeholder('float', shape = style_shape)
        net, mean_pixel = vgg.net(network_path, image)
        style_pre = np.array([vgg.subtract_mean(style, mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) # / features.size
            style_features[layer] = gram

    def content_loss(net):
        return tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER])
    
    
    def style_loss(net):
        total_style_loss = 0
        for layer in STYLE_LAYERS:
            features = net[layer]
            _, height, width, number = map(lambda i: i.value, features.get_shape())
            features = tf.reshape(features, (-1, number))
            gram = tf.matmul(tf.transpose(features), features) # / features.size
            size = height * width * number
            layer_loss = 0.5 / (size ** 2) * tf.nn.l2_loss(gram - style_features[layer])
            total_style_loss += layer_loss / len(STYLE_LAYERS)
        return total_style_loss
        
    # Optimize initial noise image achieve the stylized content image using backpropagation
    with tf.Graph().as_default():
        # create variable to hold the initial noise image with the same size of the content
        image = tf.Variable(tf.random_normal(content_shape))
        net, mean_pixel = vgg.net(network_path, image)
        
        loss = CONTENT_LOSS_WEIGHT * content_loss(net) + STYLE_LOSS_WEIGHT * style_loss(net)
        
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        
        with tf.Session() as sess:
            print('Start training')
            sess.run(tf.initialize_all_variables())
            for iteration in range(iterations):
                train_step.run()
                print(iteration)
                if iteration % 50 == 0 or iteration == iterations - 1:
                    curr_image = image.eval()
                    print(mean_pixel)
                    curr_image = vgg.add_mean(curr_image.reshape(content_shape[1:]), mean_pixel)
                    curr_image = np.clip(curr_image, 0, 255).astype(np.uint8)
                    imsave('output_%s.jpg'%iteration, curr_image)
            

content = imread('content1.jpg').astype(np.float)
style = imread('style1.jpg').astype(np.float)
stylize('imagenet-vgg-verydeep-19.mat', content, style, 1000)






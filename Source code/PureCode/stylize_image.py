import scipy.misc
import transform
import tensorflow as tf
import os
import time
import numpy as np
CONTENT_LOSS_WEIGHT = 7.5
STYLE_LOSS_WEIGHT = 1e2
TOTAL_VARIATION_LOSS_WEIGHT = 2e2
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

network_path = 'imagenet-vgg-verydeep-19.mat'

x = input()
img = scipy.misc.imread('%s.jpg'%x).astype(np.float)
content_shape = (1,)+img.shape
print(content_shape)

with tf.Graph().as_default(), tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=content_shape, name='X_content')
    #content_image_pre = vgg.subtract_mean(X_content, MEAN_PIXEL)
    
    #calculate content features
    print('Start getting content features')
    #vgg_content_net, _ = vgg.net(network_path, content_image_pre)
    #content_features[CONTENT_LAYER] = vgg_content_net[CONTENT_LAYER]
    #print(vgg_content_net)
    #get output of transform net
    transform_net = transform.net(X_content)
    
    #pass it to the vgg net to calculate losses
    #vgg_net, _ = vgg.net(network_path, vgg.subtract_mean(transform_net, MEAN_PIXEL))
    #print(vgg_net)
    
    #loss = CONTENT_LOSS_WEIGHT * content_loss(vgg_net) + STYLE_LOSS_WEIGHT * style_loss(vgg_net)
    
    #train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    print('Start training')
    #sess.run(tf.initialize_all_variables())
    
    print('Restore variables')
    saver = tf.train.Saver()
    saver.restore(sess, 'saver/w2000.ckpt')
    
    test_feed_dict = {X_content: np.reshape(img, content_shape)}
    stylized_img = transform_net.eval(feed_dict=test_feed_dict)
    #print(stylized_img[0].shape)
    print(np.max(stylized_img[0]))
    scipy.misc.imsave('%s_w2000.jpg'%x,stylized_img[0].astype(np.uint8))
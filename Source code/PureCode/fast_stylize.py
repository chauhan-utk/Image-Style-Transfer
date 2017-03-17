from __future__ import print_function
import vgg, transform
import tensorflow as tf, numpy as np, os
from operator import mul
import scipy.misc

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
CONTENT_LOSS_WEIGHT = 7.5e0
STYLE_LOSS_WEIGHT = 1e2
TV_LOSS_WEIGHT = 2e2
LEARNING_RATE = 1e-3
BATCH_SIZE = 4

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

def stylize(network_path, content_images, style_target, epochs=2):

    # Append an extra batch number dimension which is required by tensorflow
    batch_shape = (BATCH_SIZE, 256, 256, 3)
    style_shape = (1,) + style.shape
    
    # Dictionaries to hold values for content and style features
    style_features = {}
    content_features = {}
   
    def content_loss(net):
        size = reduce(mul, [i.value for i in content_features[CONTENT_LAYER].get_shape()[1:]], 1) * BATCH_SIZE ##
        return 2 * tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / size
    
    def style_loss(net):
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.batch_matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
            
        return reduce(tf.add, style_losses) / BATCH_SIZE
        
    

    def tv_loss(net):
        tv_y_size = reduce(mul, (d.value for d in net[:,1:,:,:].get_shape()[1:]), 1)
        tv_x_size = reduce(mul, (d.value for d in net[:,:,1:,:].get_shape()[1:]), 1)
        y_tv = tf.nn.l2_loss(net[:,1:,:,:] - net[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(net[:,:,1:,:] - net[:,:,:batch_shape[2]-1,:])
        return 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / BATCH_SIZE
    
    # compute style features in feedforward mode
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_subtracted_mean = vgg.subtract_mean(style_image, MEAN_PIXEL)
        net, _ = vgg.net(network_path, style_image_subtracted_mean)
        style_image_input = np.array([style_target])
        for style_layer in STYLE_LAYERS:
            features = net[style_layer].eval(feed_dict={style_image:style_image_input})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[style_layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name='X_content')
        X_content_subtracted_mean = vgg.subtract_mean(X_content, MEAN_PIXEL)
        
        print('Start getting content features')
        content_net, _ = vgg.net(network_path, X_content_subtracted_mean)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        
        transform_out = transform.net(X_content / 255.0)
        transform_out_subtracted_mean = vgg.subtract_mean(transform_out, MEAN_PIXEL)
        
        net, _ = vgg.net(network_path, transform_out_subtracted_mean)
        
        loss = CONTENT_LOSS_WEIGHT * content_loss(net) + STYLE_LOSS_WEIGHT * style_loss(net) + TV_LOSS_WEIGHT * tv_loss(transform_out)
        
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        sess.run(tf.initialize_all_variables())
        
        print('Test Saving')
        saver = tf.train.Saver()
        saver.save(sess, 'saver/test_trained_network.ckpt')
        
        NUM_EXAMPLES = len(content_images)
        for epoch in range(epochs):
            print('Epoch: %s'%epoch)
            iterations = 0
            while iterations * BATCH_SIZE < NUM_EXAMPLES:
                print('Iteration: %s'%iterations)
                curr = iterations * BATCH_SIZE
                step = curr + BATCH_SIZE
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_images[curr:step]):
                    X_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)
                iterations += 1
                feed_dict = {X_content: X_batch}
                train_step.run(feed_dict=feed_dict)
                
                if (epoch == epochs - 1 and iterations * BATCH_SIZE >= NUM_EXAMPLES) or iterations%500 == 0:
                    saver = tf.train.Saver()
                    saver.save(sess, 'saver/trained_network_%s_%s.ckpt'%(epoch,iterations))


def get_img(path, img_size=False):
    img = scipy.misc.imread(path, mode='RGB')
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

style = get_img('style_starry.jpg').astype(np.float)
content_images_paths = os.listdir('training_images/train2014')
stylize('imagenet-vgg-verydeep-19.mat', map(lambda x: os.path.join('training_images/train2014', x), content_images_paths), style, 2)






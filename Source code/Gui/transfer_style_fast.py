import transform
import transform2
import tensorflow as tf
import numpy as np

def transfer_style(img, style_name):
    content_shape = (1,)+img.shape
    
    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=content_shape, name='X_content')
        if style_name != 'starry_night_diff':
            transform_net = transform.net(X_content)
        else:
            transform_net = transform2.net(X_content)
        print('Restore model')
        saver = tf.train.Saver()
        saver.restore(sess, 'saver/%s.ckpt'%style_name)
        test_feed_dict = {X_content: np.reshape(img, content_shape)}
        stylized_img = transform_net.eval(feed_dict=test_feed_dict)
        return stylized_img[0].astype(np.uint8)
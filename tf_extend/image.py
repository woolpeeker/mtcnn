
import tensorflow as tf
import tf_extend as tfe

'''
TODO: shape problem
def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.cond(pred = tf.equal(tf.rank(image),3),
                        true_fn = lambda: tf.expand_dims(image, 0),
                        false_fn= lambda: image)
        image = tf.image.resize_images(image, size, method, align_corners)
        return image
'''

def resize_image_no_batch(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=True, scope='resize_iamge_no_batch'):
    with tf.name_scope(scope):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.squeeze(image, 0)
        return image

def img_shape(img, dtype=tf.int32, scope='img_shape'):
    with tf.name_scope(scope):
        with tf.control_dependencies([tf.assert_rank_in(img, (3, 4))]):
            result = tf.cond(pred = tf.equal(tf.rank(img),3),
                             true_fn=lambda: tf.shape(img)[:2],
                             false_fn=lambda: tf.shape(img)[1:3])
            return tf.cast(result, dtype)


def pyramid_scale(image, scale=0.7, min_len=12, scope='pyramid_scale'):
    with tf.name_scope(scope):
        im_arr=tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=[None,None,3],
                              infer_shape=False, clear_after_read=False)
        im_arr = im_arr.write(0,image)
        def body(i, im_arr):
            s = tf.pow(scale,tf.cast(i, tf.float32))
            target_shape = tf.cast(s * tfe.img_shape(image, tf.float32), tf.int32)
            scaled_im = tf.image.resize_bilinear(tf.expand_dims(image,0), target_shape,align_corners=True)
            im_arr = im_arr.write(i,tf.squeeze(scaled_im, axis=0))
            return i+1, im_arr
        i=1
        i, im_arr = tf.while_loop(loop_vars=[i, im_arr],
                                  cond=lambda i,im_arr: tf.reduce_min(tfe.img_shape(im_arr.read(i-1)))>min_len,
                                  body=body,
                                  parallel_iterations=8)
    return im_arr
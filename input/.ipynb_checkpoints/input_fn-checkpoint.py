import os
import tensorflow as tf

from datasets.widerface import get_dataset
from input.preprocess import preprocess_for_train_pnet, preprocess_choose_single_bbox, preprocess_random_scale
import tf_extend as tfe

from nets.pnet import PNet
pnet=PNet()

def input_pnet_train_fn():
    tfname = 'datasets/wider_train_cropped80_scaleTo15_repeat50.tfrecord'
    tfrecordObj = tfe.ImageBboxTfrecord()
    dataset = tfrecordObj.read(tfname,n_thread=16)
    def bboxes_encode(x):
        cls, loc = pnet.train_anchor_obj_fn().bboxes_encode(x['bboxes'])
        return {'image': x['image'],
                'gt_cls': cls,
                'gt_loc': loc}
    dataset = dataset.map(bboxes_encode, num_parallel_calls=16)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(None)
    return dataset

def input_pnet_gen_samples_fn():
    dataset = get_dataset('datasets/widerface/wider_train_*.tfrecord')
    def bboxes_encode(x):
        return {'image': tf.cast(x['image'], tf.float32)}
    dataset = dataset.map(bboxes_encode, num_parallel_calls=8)
    dataset = dataset.prefetch(10)
    return dataset

def input_rnet_train_fn():
    wider_dataset = get_dataset('datasets/widerface/wider_train_*.tfrecord',)
    pnet_dataset = tfe.Bboxes_tfrecord().read('gen_samples/pnet/pnet_samples.tfrecord')
    dataset = tf.data.Dataset.zip((wider_dataset, pnet_dataset))
    def fn1(x1, x2):
        image = tf.cast(x1['image'], tf.float32)
        bboxes = x1['bboxes']
        sample_bboxes = x2['bboxes']
        sample_cls, sample_loc = tfe.encode_pred_bboxes(sample_bboxes, bboxes)
        crop_bboxes = tfe.convert_bboxes_to_float(sample_bboxes, tfe.img_shape(image))
        sample_imgs = tf.image.crop_and_resize(tf.expand_dims(image,0), crop_bboxes,
                                               tf.zeros_like(sample_cls, tf.int32), (24,24))
        #image_bboxes = tf.image.draw_bounding_boxes(image, tf.expand_dims(crop_bboxes,0))
        #tiled_image = tf.tile(image_bboxes,
        #                      tf.concat([tf.shape(sample_cls), tf.ones(3,dtype=tf.int32)], axis=0))
        return sample_imgs, sample_cls, sample_loc

    def flat_map_fn(images, cls, loc):
        return tf.data.Dataset.from_tensor_slices({'image':images, 'gt_cls':cls, 'gt_loc':loc})

    dataset = dataset.map(fn1,num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.flat_map(flat_map_fn)
    dataset = dataset.batch(1024)
    dataset = dataset.prefetch(None)
    return dataset

def input_rnet_gen_samples_fn():
    wider_dataset = get_dataset('datasets/widerface/wider_train_*.tfrecord',)
    pnet_dataset = tfe.Bboxes_tfrecord().read('gen_samples/pnet/pnet_samples.tfrecord')
    dataset = tf.data.Dataset.zip((wider_dataset, pnet_dataset))
    def fn1(x1, x2):
        image = tf.cast(x1['image'], tf.float32)
        image = tf.expand_dims(image, 0)
        gt_bboxes = x1['bboxes']
        sample_bboxes = x2['bboxes']
        float_sample_bboxes = tfe.convert_bboxes_to_float(sample_bboxes, tfe.img_shape(image))
        sample_imgs = tf.image.crop_and_resize(image, float_sample_bboxes,
                                               tf.zeros([tf.shape(float_sample_bboxes)[0]], tf.int32), (24,24))
        image = tf.image.draw_bounding_boxes(image, tf.expand_dims(float_sample_bboxes,0))
        image = tf.squeeze(image,0)
        return {'image': image,
                'gt_bboxes': gt_bboxes,
                'sample_bboxes': sample_bboxes,
                'sample_imgs': sample_imgs}

    dataset = dataset.map(fn1,num_parallel_calls=8)
    dataset = dataset.prefetch(None)
    return dataset

def input_onet_train_fn():
    wider_dataset = get_dataset('datasets/widerface/wider_train_*.tfrecord',)
    pnet_dataset = tfe.Bboxes_tfrecord().read('gen_samples/rnet/rnet_samples.tfrecord')
    dataset = tf.data.Dataset.zip((wider_dataset, pnet_dataset))
    def fn1(x1, x2):
        image = tf.cast(x1['image'], tf.float32)
        bboxes = x1['bboxes']
        sample_bboxes = x2['bboxes']
        sample_cls, sample_loc = tfe.encode_pred_bboxes(sample_bboxes, bboxes)
        crop_bboxes = tfe.convert_bboxes_to_float(sample_bboxes, tfe.img_shape(image))
        sample_imgs = tf.image.crop_and_resize(tf.expand_dims(image,0), crop_bboxes,
                                               tf.zeros([tf.shape(crop_bboxes)[0]], tf.int32), (48,48))
        #image_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image,0), tf.expand_dims(crop_bboxes,0))
        #tiled_image = tf.tile(image_bboxes,
        #                      tf.concat([tf.shape(sample_cls), tf.ones(3,dtype=tf.int32)], axis=0))
        return sample_imgs, sample_cls, sample_loc

    def flat_map_fn(images, cls, loc):
        return tf.data.Dataset.from_tensor_slices({'image':images, 'gt_cls':cls, 'gt_loc':loc})

    dataset = dataset.map(fn1,num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.flat_map(flat_map_fn)
    dataset = dataset.batch(512)
    dataset = dataset.prefetch(None)
    return dataset

def input_onet_gen_samples_fn():
    wider_dataset = get_dataset('datasets/widerface/wider_train_*.tfrecord', )
    pnet_dataset = tfe.Bboxes_tfrecord().read('gen_samples/rnet/rnet_samples.tfrecord')
    dataset = tf.data.Dataset.zip((wider_dataset, pnet_dataset))

    def fn1(x1, x2):
        image = tf.cast(x1['image'], tf.float32)
        sample_bboxes = x2['bboxes']
        crop_bboxes = tfe.convert_bboxes_to_float(sample_bboxes, tfe.img_shape(image))
        sample_imgs = tf.image.crop_and_resize(tf.expand_dims(image, 0), crop_bboxes,
                                               tf.zeros([tf.shape(crop_bboxes)[0]], tf.int32),
                                               (48,48))
        return {'sample_bboxes': sample_bboxes,
                'sample_imgs': sample_imgs,}
    dataset = dataset.map(fn1, num_parallel_calls=8)
    dataset = dataset.prefetch(None)
    return dataset
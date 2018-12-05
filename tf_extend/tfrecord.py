import tensorflow as tf
import glob

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

class Bboxes_tfrecord:
    def write(self, tf_fname, samples, verbosity=False):
        with tf.python_io.TFRecordWriter(tf_fname) as tf_writer:
            count=0
            for sample in samples:
                count+=1
                if verbosity and count % 100 == 0:
                    print('write %d samples'%count)
                bboxes = sample['bboxes'].flatten().tolist()
                features = tf.train.Features(feature={'bboxes': float_feature(bboxes)})
                example = tf.train.Example(features=features)
                tf_writer.write(example.SerializeToString())
            tf.logging.info('Bboxes_tfrecord write finished with num_samples:%d' % count)

    def read(self, tf_fname):
        fnames = glob.glob(tf_fname)
        if not fnames:
            raise Exception('%s do not exist' % tf_fname)
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=4)
        dataset = dataset.map(self.parser, num_parallel_calls=4)
        return dataset

    def parser(self, record):
        keys_to_features = {'bboxes': tf.VarLenFeature(dtype=tf.float32)}
        features = tf.parse_single_example(record, features=keys_to_features)
        bboxes = tf.sparse_tensor_to_dense(features['bboxes'])
        bboxes = tf.reshape(bboxes, [-1, 4])
        return {'bboxes': bboxes}


class ImageBboxTfrecord:
    def write(self, fname, sample_gen):
        #sample_gen is a generator
        with tf.python_io.TFRecordWriter(fname) as tf_writer:
            for sample in sample_gen:
                example = self._convert_to_example(sample)
                tf_writer.write(example.SerializeToString())

    def _convert_to_example(self, sample):
        image = sample['image']
        bboxes = sample['bboxes'].reshape([-1]).tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(image),
            'bboxes': float_feature(bboxes)
        }))
        return example

    def read(self, fname, n_thread=8):
        fnames = glob.glob(fname)
        if not fnames:
            raise Exception('%s do not exist' % fnames)
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=n_thread)
        dataset = dataset.map(self.parser, num_parallel_calls=n_thread)
        return dataset

    def parser(self, record):
        keys_to_features = {'image': tf.FixedLenFeature((), tf.string),
                            'bboxes': tf.VarLenFeature(dtype=tf.float32)}
        features = tf.parse_single_example(record, features=keys_to_features)
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.to_float(image)
        bboxes = tf.sparse_tensor_to_dense(features['bboxes'])
        bboxes = tf.reshape(bboxes, [-1, 4])
        return {'image': image, 'bboxes':bboxes}
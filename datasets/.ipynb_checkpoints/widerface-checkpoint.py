import tensorflow as tf
import tensorflow.contrib.slim as slim

import glob


def get_wider_demo1_data(is_training_data=True):
    data_sources = "datasets/wider_demo1/wider_demo*.tfrecord"
    num_samples = DATASET_SIZE['wider_demo1']
    return get_dataset(data_sources), num_samples

def get_wider_train_data(shard=None,is_training_data=True):
    data_sources = "datasets/wider_face/wider_train_*.tfrecord"
    num_samples = DATASET_SIZE['wider_train']
    return get_dataset(data_sources,shard=shard), num_samples


ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

DATASET_SIZE = {
    'wider_val': 3226,
    'wider_test': 5011,
    'wider_train': 12880,
    'wider_demo1': 1,
}

DATASET_FN = {
    'wider_train': get_wider_train_data,
    'wider_demo1': get_wider_demo1_data,
}

def get_dataset(data_sources, shard=None):
    fnames = glob.glob(data_sources)
    dataset = tf.data.TFRecordDataset(fnames,num_parallel_reads=8)

    def parser(record):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value='000000'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64)
        }
        features = tf.parse_single_example(record, features=keys_to_features)
        image = tf.image.decode_jpeg(features['image/encoded'],channels=3)
        bboxes = [features['image/object/bbox/'+x] for x in ['ymin','xmin','ymax','xmax']]
        bboxes = [tf.sparse_tensor_to_dense(x) for x in bboxes]
        bboxes = tf.stack(bboxes,axis=-1)
        bboxes = tf.cast(bboxes, tf.float32) #tf is 1 based bboxes and widerface is 0 based
        return {"image": image, 'bboxes':bboxes}
    if shard is not None:
        dataset=dataset.shard(shard,0)
    dataset = dataset.map(parser,num_parallel_calls=16)
    return dataset

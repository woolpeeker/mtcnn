
import tensorflow as tf
import tf_extend as tfe
import re

imageListFile = "/home/luojiapeng/datasets/fddb/ellipseList.txt"

def parser(record):
    raw = tf.io.read_file(record)
    image = tf.image.decode_jpeg(raw, channels=3)
    return {'image':image}

def get_fddb(n_thread=16):
    imageList=[]
    for line in open(imageListFile,'r').readlines():
        line=line.rstrip()
        if re.search('img', line):
            imageList.append(line+'.jpg')
    dataset = tf.data.Dataset.from_tensor_slices((imageList,))
    dataset = dataset.map(parser,num_parallel_calls=n_thread)
    return dataset
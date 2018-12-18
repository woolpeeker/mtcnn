
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

import os, re

import tensorflow as tf
import tf_extend as tfe
import numpy as np

from nets.mtcnn import MTCNN

fddb_imgs_file = '/home/luojiapeng/datasets/fddb/imgList.txt'
fddb_anno_file = '/home/luojiapeng/datasets/fddb/ellipseList.txt'
fddb_base_dir = '/home/luojiapeng/datasets/fddb'
mtcnn_obj = MTCNN()


def get_fddb_imgs():
    anno_file = open(fddb_imgs_file, 'r')
    fnames = []
    for line in anno_file.readlines():
        line = line.rstrip()
        if line:
            fnames.append(line)
    return fnames

def get_fddb():
    fnames = get_fddb_imgs()
    fnames = [os.path.join(fddb_base_dir, x)+'.jpg' for x in fnames]
    dataset = tf.data.Dataset.from_tensor_slices((fnames,))
    def fn(fname):
        raw_img = tf.read_file(fname)
        image = tf.image.decode_jpeg(raw_img, channels=3)
        image = tf.cast(image, tf.float32)
        return {'fname': fname, 'image': image}
    dataset = dataset.map(fn, num_parallel_calls=8)
    return dataset

def save_fddb_result(outFile, fnames, result):
    with open(outFile,'w') as fid:
        for i in range(len(fnames)):
            fname = fnames[i]
            bboxes, scores = result[i]['bboxes'], result[i]['scores']
            assert len(bboxes) == len(scores)

            fid.write(fname+'\n')
            if bboxes is None:
                fid.write(str(1) + '\n')
                fid.write('%d %d %d %d %f\n' % (0, 0, 0, 0, 0.01))
                continue
            else:
                fid.write(str(len(bboxes)) + '\n')
                for _i in range(len(scores)):
                    s, b =scores[_i], bboxes[_i]
                    b=[int(np.round(x)) for x in b]
                    fid.write('%d %d %d %d %.3f\n' % (b[1], b[0], b[3] - b[1] + 1, b[2] - b[0] + 1, s))
                if i % 100 == 0 and i:
                    print(i)
        fid.close()

def eval_on_fddb():
    output_dir = tfe.get_checkpoint_dir('result', 'fddb_quantized')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outFile = os.path.join(output_dir, 'detectBboxes.txt')
    params = {
        'pnet_model_dir': 'checkpoints/pnet',
        'pnet_thres': 0.3,
        'rnet_model_dir': 'checkpoints/rnet',
        'rnet_thres': 0.3,
        'onet_model_dir': 'checkpoints/onet',
        'onet_thres': 0.2
    }
    p_res, r_res, o_res = mtcnn_obj.predict(get_fddb, params)
    fnames = get_fddb_imgs()
    save_fddb_result(outFile, fnames, o_res)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    eval_on_fddb()
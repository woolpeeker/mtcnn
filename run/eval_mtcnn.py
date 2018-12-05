
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import os, re

import tensorflow as tf
import tf_extend as tfe

from nets.mtcnn import MTCNN

wider_anno_file = '/home/luojiapeng/datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt'
wider_base_dir = '/home/luojiapeng/datasets/widerface/WIDER_val/images'

mtcnn_obj = MTCNN()

def get_wider_imgs():
    anno_file = open(wider_anno_file, 'r')
    fnames = []
    for line in anno_file.readlines():
        line = line.rstrip()
        if re.match('^.*\.jpg$', line):
            fnames.append(line)
    return fnames


def get_wider_val():
    fnames = get_wider_imgs()
    fnames = [os.path.join(wider_base_dir, x) for x in fnames]
    dataset = tf.data.Dataset.from_tensor_slices((fnames,))

    def fn(fname):
        raw_img = tf.read_file(fname)
        image = tf.image.decode_jpeg(raw_img, channels=3)
        image = tf.cast(image, tf.float32)
        return {'fname': fname, 'image': image}

    dataset = dataset.map(fn, num_parallel_calls=8)
    return dataset

def save_wider_result(output_dir, fnames, result):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    current_event = ''
    for i in range(len(fnames)):
        fname = fnames[i]
        bboxes, scores = result[i]['bboxes'], result[i]['scores']
        assert len(bboxes) == len(scores)
        event = fname.split('/')[0]
        if current_event != event:
            current_event = event
            save_path = os.path.join(output_dir, current_event)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print('current path:', current_event)

        out_fname = fname.split('.jpg')[0]
        out_fname = os.path.join(output_dir, out_fname + '.txt')
        fid = open(out_fname, 'w')
        fid.write(fname.split('/')[-1] + '\n')
        if bboxes is None:
            fid.write(str(1) + '\n')
            fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.01))
            continue
        else:
            fid.write(str(len(bboxes)) + '\n')
            for _i in range(len(scores)):
                s, b =scores[_i], bboxes[_i]
                fid.write('%.2f %.2f %.2f %.2f %.2f\n' % (b[1], b[0], b[3] - b[1] + 1, b[2] - b[0] + 1, s))

            fid.close()
            if i % 100 == 0 and i:
                print(i)

def eval_on_wider():
    output_dir = tfe.get_checkpoint_dir('result', 'pnet_mse_l2t3_0_rnet_l2t3_0_onet_pin_1')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    params = {
        'pnet_model_dir': 'checkpoints/pnet_mse_l2t3_0',
        'pnet_thres': 0.3,
        'rnet_model_dir': 'checkpoints/rnet_l2t3_0',
        'rnet_thres': 0.3,
        'onet_model_dir': 'checkpoints/onet_pin_1',
        'onet_thres': 0.3
    }
    p_res, r_res, o_res = mtcnn_obj.predict(get_wider_val, params)
    fnames = get_wider_imgs()
    save_wider_result(os.path.join(output_dir, 'pnet'), fnames, p_res)
    save_wider_result(os.path.join(output_dir, 'rnet'), fnames, r_res)
    save_wider_result(os.path.join(output_dir, 'onet'), fnames, o_res)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    eval_on_wider()
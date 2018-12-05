import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

import tf_extend as tfe

from nets.onet import ONet
import input.input_fn as input_fn


pnet_tfname = 'checkpoints/pnet_mse_l2t3_0/pnet.tfrecord'
rnet_tfname = 'checkpoints/rnet_l2t3_0/rnet.tfrecord'
train_input_fn = lambda: input_fn.input_onet_train_fn(tfnames=[pnet_tfname, rnet_tfname])
#gen_samples_fn = lambda: input_fn.input_onet_gen_samples_fn(tfname=rnet_tfname)

onet=ONet()

def train_onet():
    output_dir = tfe.get_checkpoint_dir(fold='onet_pin')
    runConfig = tf.estimator.RunConfig(model_dir=output_dir,
                                       save_checkpoints_steps=1e4)
    params = {'output_dir': output_dir,
              'thres': 0.3}
    classifier = tf.estimator.Estimator(model_fn=onet.model_fn,
                                        params=params,
                                        config=runConfig)
    classifier.train(train_input_fn, max_steps=3e5)
    return

def gen_samples():
    model_dir = 'checkpoints/onet_0'
    runConfig = tf.estimator.RunConfig(model_dir=model_dir)
    params = {'thres': 0.3}
    classifier = tf.estimator.Estimator(model_fn=onet.model_fn,
                                        params=params,
                                        config=runConfig)
    result = classifier.predict(input_fn.input_onet_gen_samples_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    mode = 'train'

    if mode == 'train':
        train_onet()
    if mode == 'gen_samples':
        gen_samples()
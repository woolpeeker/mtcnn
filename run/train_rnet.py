import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import tf_extend as tfe

from nets.rnet import RNet
import input.input_fn as input_fn

tf.logging.set_verbosity(tf.logging.DEBUG)


rnet=RNet()

tfname = 'checkpoints/pnet_mse_l2t3_0/pnet.tfrecord'
train_input_fn = lambda: input_fn.input_rnet_train_fn(tfname=tfname)
gen_samples_fn = lambda: input_fn.input_rnet_gen_samples_fn(tfname=tfname)

def train():
    output_dir = tfe.get_checkpoint_dir(fold='rnet_l2t3')
    runConfig = tf.estimator.RunConfig(model_dir=output_dir,
                                       save_checkpoints_steps=2000)
    params = {'output_dir': output_dir,
              'thres': 0.3}
    classifier = tf.estimator.Estimator(model_fn=rnet.model_fn,
                                        params=params,
                                        config=runConfig)
    classifier.train(train_input_fn, max_steps=3e5)
    return

def gen_samples():
    model_dir = 'checkpoints/rnet_l2t3_0'
    runConfig = tf.estimator.RunConfig(model_dir=model_dir)
    params = {'gen_samples': True,
              'thres': 0.3}
    classifier = tf.estimator.Estimator(model_fn=rnet.model_fn,
                                        params=params,
                                        config=runConfig)
    results = classifier.predict(gen_samples_fn)

    tfrecord_obj = tfe.Bboxes_tfrecord()
    tf_fname = os.path.join(model_dir, 'rnet.tfrecord')
    tfrecord_obj.write(tf_fname, results, verbosity=True)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    mode = 'gen_samples'

    if mode == 'train':
        train()
    if mode == 'gen_samples':
        gen_samples()
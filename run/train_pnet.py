import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tf_extend as tfe

from nets.pnet import PNet
import input.input_fn as input_fn
pnet = PNet()


def train():
    output_dir = tfe.get_checkpoint_dir(fold='pnet_mse_l2t3')
    runConfig = tf.estimator.RunConfig(model_dir=output_dir,
                                       save_checkpoints_steps=5e3)
    params = {'output_dir': output_dir,
              'thres': 0.3}
    classifier = tf.estimator.Estimator(model_fn=pnet.model_fn,
                                        params=params,
                                        config=runConfig)
    classifier.train(input_fn.input_pnet_train_fn, max_steps=5e5)


def gen_samples():
    model_dir = 'checkpoints/pnet_mse_l2t3_0'
    runConfig = tf.estimator.RunConfig(model_dir=model_dir)
    params = {'gen_samples': True,
              'thres': 0.3}
    classifier = tf.estimator.Estimator(model_fn=pnet.model_fn,
                                        params=params,
                                        config=runConfig)
    results = classifier.predict(input_fn.input_pnet_gen_samples_fn)

    tfrecord_obj = tfe.Bboxes_tfrecord()
    tf_fname = os.path.join(model_dir,'pnet.tfrecord')
    tfrecord_obj.write(tf_fname, results, verbosity=True)
    return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    mode = 'train'

    if mode == 'train':
        train()
    if mode == 'gen_samples':
        gen_samples()

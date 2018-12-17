import tensorflow as tf

import tf_extend as tfe
import numpy as np

from nets.pnet import PNet
from nets.rnet import RNet
from nets.onet import ONet


class MTCNN:
    def predict(self, input_fn, params):
        #input_fn should be a dataset: {'fname','image'}
        pnet_result, rnet_result, onet_result = None, None, None

        pnet = PNet()
        pnet_model_dir = params['pnet_model_dir']
        pnet_runConfig = tf.estimator.RunConfig(model_dir=pnet_model_dir)
        pnet_params = {'thres': params['pnet_thres']}
        classifier = tf.estimator.Estimator(model_fn=pnet.model_fn,
                                            params=pnet_params,
                                            config=pnet_runConfig)
        pnet_input_fn = input_fn
        hooks=[tfe.quantizeHook()]
        pnet_result = classifier.predict(pnet_input_fn, hooks=hooks)
        pnet_result = list(pnet_result)

        if 'rnet_model_dir' in params and params['rnet_model_dir']:
            rnet = RNet()
            rnet_model_dir = params['rnet_model_dir']
            rnet_runConfig = tf.estimator.RunConfig(model_dir=rnet_model_dir)
            rnet_params = {'thres': params['rnet_thres']}
            classifier = tf.estimator.Estimator(model_fn=rnet.model_fn,
                                                params=rnet_params,
                                                config=rnet_runConfig)
            rnet_input_fn = lambda: self.convert_result_to_dataset(input_fn,pnet_result, (24,24))
            hooks = [tfe.quantizeHook()]
            rnet_result = classifier.predict(rnet_input_fn, hooks=hooks)
            rnet_result = list(rnet_result)

        if 'onet_model_dir' in params and params['onet_model_dir']:
            onet = ONet()
            onet_model_dir = params['onet_model_dir']
            onet_runConfig = tf.estimator.RunConfig(model_dir=onet_model_dir)
            onet_params = {'thres': params['onet_thres']}
            classifier = tf.estimator.Estimator(model_fn=onet.model_fn,
                                                params=onet_params,
                                                config=onet_runConfig)
            onet_input_fn = lambda: self.convert_result_to_dataset(input_fn, rnet_result, (48,48))
            hooks = [tfe.quantizeHook()]
            onet_result = classifier.predict(onet_input_fn, hooks=hooks)
            onet_result = list(onet_result)
        return pnet_result, rnet_result, onet_result

    def convert_result_to_dataset(self, input_fn, net_result, size):
        net_result = tuple([x['bboxes'] for x in net_result])
        def gen(net_result = net_result):
            for x in net_result:
                yield x
        net_result = tf.data.Dataset.from_generator(generator = lambda: gen(),
                                                    output_types = tf.float32,
                                                    output_shapes=[None, 4])
        dataset = tf.data.Dataset.zip((input_fn(), net_result))
        def fn(x1,x2):
            image = x1['image']
            bboxes = x2
            crop_bboxes = tfe.convert_bboxes_to_float(bboxes, tfe.img_shape(image))
            sample_imgs = tf.image.crop_and_resize(tf.expand_dims(image, 0), crop_bboxes,
                                                   tf.zeros([tf.shape(bboxes)[0]], tf.int32),
                                                   size)
            return {'sample_imgs':sample_imgs,
                    'sample_bboxes':bboxes}
        dataset = dataset.map(fn, num_parallel_calls=8)
        return dataset

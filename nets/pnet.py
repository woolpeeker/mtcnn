import tensorflow as tf
import tensorflow.contrib.slim as slim
ModeKeys=tf.estimator.ModeKeys

import numpy as np

import tf_extend as tfe


class PNet:
    def __init__(self):
        self.train_image_shape = (80, 80)
        self.anchor_size = 15
        self.anchor_step = 4
        self.train_anchor_obj_fn = lambda: tfe.Anchor(anchor_shape=(20, 20),
                                                      anchor_size=self.anchor_size,
                                                      anchor_step=self.anchor_step,
                                                      align_corner=False)

    def model_fn(self, features, labels, mode, params=None, config=None):
        if mode == ModeKeys.TRAIN:
            return self.model_train_fn(mode, features, params)
        elif mode == ModeKeys.PREDICT:
            return self.model_gen_samples_fn(mode,features, params)


    def pnet(self, images, is_training=False, scope='pnet'):
        with tf.name_scope(scope, 'pnet', values=[images]):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='SAME'),\
                slim.arg_scope([slim.batch_norm],
                               is_training=is_training):
                    images = images/255-0.5
                    net = slim.conv2d(images, num_outputs=10, kernel_size=[3, 3], stride=2)
                    net = tfe.fake_quantize(net)
                    net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1)
                    net = tfe.fake_quantize(net)
                    net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=2)
                    net = tfe.fake_quantize(net)
                    net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1)
                    net = tfe.fake_quantize(net)
                    pred_logit = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1,
                                           activation_fn=None, normalizer_fn=None)
                    pred_logit = tfe.fake_quantize(pred_logit)
                    pred_cls = slim.softmax(pred_logit)[...,1]
                    pred_cls = tfe.fake_quantize(pred_cls)
                    pred_loc = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1,
                                           activation_fn=None, normalizer_fn=None)
                    pred_loc = tfe.fake_quantize(pred_loc)
                    predictions = {'pred_cls': pred_cls,
                                   'pred_logit':pred_logit,
                                   'pred_loc': pred_loc}
        return predictions

    def model_train_fn(self, mode, features, params):
        images = features['image']
        thres = params['thres']
        predictions = self.pnet(images, is_training=True)
        images_shape = tf.shape(images)[1:3]
        gt_cls, gt_loc = features['gt_cls'], features['gt_loc']
        pred_cls, pred_loc = predictions['pred_cls'], predictions['pred_loc']

        optimizer = tf.train.AdamOptimizer(tfe.lr_plateau_decay(lr=0.001, decay=0.9999, min_lr=1e-6))
        step = tf.train.get_or_create_global_step()
        pnet_loss = self.loss_fn(gt_cls, gt_loc, predictions, thres)
        regular_loss = tf.add_n(tf.losses.get_regularization_losses())
        total_loss = pnet_loss['total_loss'] + regular_loss
        minimize_op = optimizer.minimize(total_loss, step)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_op)

        # summary
        anchor_obj = self.train_anchor_obj_fn()
        gt_bboxes, _ = anchor_obj.bboxes_decode(gt_cls, gt_loc, 0.001)
        gt_bboxes = tfe.convert_bboxes_to_float(gt_bboxes, images_shape)
        _, gt_bboxes = tfe.bboxes_nms_batch(tf.ones(tf.shape(gt_bboxes)[:2]), gt_bboxes)
        images_gt_bboxes = tf.image.draw_bounding_boxes(images, gt_bboxes)

        pred_bboxes, pred_scores = anchor_obj.bboxes_decode(pred_cls, pred_loc, thres)
        pred_bboxes = tfe.convert_bboxes_to_float(pred_bboxes, images_shape)
        nms_scores, nms_bboxes = tfe.bboxes_nms_batch(pred_scores, pred_bboxes)
        images_pred_bboxes = tf.image.draw_bounding_boxes(images, nms_bboxes)

        tf.summary.image('images_gt_bboxes', images_gt_bboxes)
        tf.summary.image('images_pred_bboxes', images_pred_bboxes)
        tf.summary.image('gt_cls', tf.expand_dims(gt_cls,-1))
        tf.summary.image('pred_cls', tf.expand_dims(pred_cls,-1))
        self.bboxes_metrics_fn(gt_bboxes, nms_bboxes, thres)
        self.metrics_fn(gt_cls,pred_cls, thres)
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            tf.summary.histogram(var.name, var)
        return tf.estimator.EstimatorSpec(mode, predictions, total_loss, train_op)

    def model_gen_samples_fn(self, mode, features, params):
        image = features['image']
        thres = params['thres']
        sample = self.gen_sample(image, thres)
        hooks = [tf.train.LoggingTensorHook([], every_n_iter=100)]
        return tf.estimator.EstimatorSpec(mode, sample, prediction_hooks=hooks)

    def gen_sample(self, image, thres):
        im_arr = tfe.pyramid_scale(image, scale=0.8, min_len=10)
        pred_scores_arr = tf.TensorArray(tf.float32, size=0, element_shape=[],
                                    dynamic_size=True, clear_after_read=False)
        #this is float bboxes which is invariant to scale
        pred_bboxes_arr = tf.TensorArray(tf.float32, size=0, element_shape=[4],
                                    dynamic_size=True, clear_after_read=False)
        def body(i, pred_scores_arr, pred_bboxes_arr, im_arr=im_arr):
            im = im_arr.read(i)
            anchor_obj = tfe.Anchor(tf.ceil(tfe.img_shape(im) / self.anchor_step),
                                    anchor_size=self.anchor_size,
                                    anchor_step=self.anchor_step,
                                    align_corner=False)
            im = tf.expand_dims(im, 0)

            pred = self.pnet(im,is_training=False)
            pred_cls, pred_loc = pred['pred_cls'], pred['pred_loc']
            pred_bboxes, pred_scores = anchor_obj.bboxes_decode(pred_cls, pred_loc, match_thres=thres)
            pred_bboxes = tf.squeeze(pred_bboxes, 0)
            pred_scores = tf.squeeze(pred_scores, 0)
            pred_bboxes_float = tfe.convert_bboxes_to_float(pred_bboxes, tfe.img_shape(im))

            arr_size = pred_scores_arr.size()
            indices = tf.range(arr_size, arr_size + tf.shape(pred_scores)[0])
            pred_scores_arr = pred_scores_arr.scatter(indices, pred_scores)
            pred_bboxes_arr = pred_bboxes_arr.scatter(indices, pred_bboxes_float)
            return i+1, pred_scores_arr, pred_bboxes_arr
        i = 0
        i, pred_scores_arr, pred_bboxes_arr= \
            tf.while_loop(cond=lambda i, _s, _b: i < im_arr.size(),
                          loop_vars=[i, pred_scores_arr, pred_bboxes_arr],
                          body=body,
                          parallel_iterations=8,back_prop=False)
        pred_scores_all = pred_scores_arr.stack()
        pred_bboxes_all = pred_bboxes_arr.stack()
        nms_idx = tf.image.non_max_suppression(pred_bboxes_all, pred_scores_all, iou_threshold=0.5,
                                                 max_output_size=400, score_threshold=thres)
        nms_scores = tf.expand_dims(tf.gather(pred_scores_all, nms_idx), 0)
        nms_bboxes = tf.gather(pred_bboxes_all, nms_idx)
        #image_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.expand_dims(nms_bboxes,0))
        nms_bboxes = tfe.convert_bboxes_to_int(nms_bboxes, tfe.img_shape(image))
        nms_bboxes = tf.expand_dims(nms_bboxes, axis=0)
        samples = {'bboxes':nms_bboxes, 'scores':nms_scores}
        return samples

    def loss_fn(self, gt_cls, gt_loc, predictions, thres, scope='loss_fn'):
        with tf.name_scope(scope):
            pred_cls = predictions['pred_cls']
            pred_logit = predictions['pred_logit']
            pred_loc = predictions['pred_loc']
            gt_cls = tf.reshape(gt_cls, [-1])
            gt_loc = tf.reshape(gt_loc, [-1, 4])
            pred_cls = tf.reshape(pred_cls, [-1])
            pred_logit = tf.reshape(pred_logit, [-1,2])
            pred_loc = tf.reshape(pred_loc, [-1, 4])

            pmask = gt_cls > thres
            nmask = gt_cls < 0.1
            gt_labels = tf.to_int32(pmask)
            num_pos = tf.reduce_sum(tf.to_int32(pmask))
            num_neg = tf.reduce_sum(tf.to_int32(nmask))

            pos_loss = tf.losses.mean_squared_error(predictions=tf.boolean_mask(pred_cls, pmask),
                                                    labels=tf.boolean_mask(gt_cls, pmask))
            neg_loss = tf.losses.mean_squared_error(predictions=tf.boolean_mask(pred_cls, nmask),
                                                    labels=tf.boolean_mask(gt_cls, nmask))
            #pos_loss = tf.losses.sparse_softmax_cross_entropy(logits=tf.boolean_mask(pred_logit, pmask),
            #                                                  labels=tf.boolean_mask(gt_labels, pmask))
            #neg_loss = tf.losses.sparse_softmax_cross_entropy(logits=tf.boolean_mask(pred_logit, nmask),
            #                                                  labels=tf.boolean_mask(gt_labels, nmask))
            loc_loss = tf.losses.mean_squared_error(predictions=tf.boolean_mask(pred_loc, pmask),
                                                    labels=tf.boolean_mask(gt_loc, pmask))
            cls_loss = 2 * pos_loss + neg_loss
            total_loss = cls_loss + loc_loss
            tf.summary.scalar('pos_loss', tf.reduce_mean(pos_loss))
            tf.summary.scalar('neg_loss', tf.reduce_mean(neg_loss))
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('loc_loss', loc_loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('num_pos', num_pos)
            tf.summary.scalar('num_neg', num_neg)
            return {
                'cls_loss': cls_loss,
                'loc_loss': loc_loss,
                'total_loss': total_loss
            }

    def metrics_fn(self, gt_cls, pred_cls, thres, scope='metrics_fn'):
        with tf.name_scope(scope):
            gt_cls = tf.reshape(gt_cls, [-1])
            pred_cls = tf.reshape(pred_cls, [-1])
            gt_cls = gt_cls > thres
            pred_cls = pred_cls > thres
            prec = tfe.precision(gt_cls, pred_cls)
            recall = tfe.recall(gt_cls, pred_cls)
            tf.summary.scalar('precision', prec)
            tf.summary.scalar('recall', recall)

    def bboxes_metrics_fn(self, gt_bboxes, pred_bboxes, thres, scope='bboxes_metrics_fn'):
        with tf.name_scope(scope):
            prec = tfe.bboxes_precision(gt_bboxes, pred_bboxes, thres)
            recall = tfe.bboxes_recall(gt_bboxes, pred_bboxes, thres)
            tf.summary.scalar('bboxes_precision', prec)
            tf.summary.scalar('bboxes_recall', recall)

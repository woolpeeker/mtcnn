import tensorflow as tf
import tensorflow.contrib.slim as slim

import tf_extend as tfe

ModeKeys=tf.estimator.ModeKeys

class RNet:
    def __init__(self):
        self.input_shape = (24, 24)
    def model_fn(self, features, labels, mode, params=None, config=None):
        """features should has at least three keys: image, gt_cls, gt_loc"""
        if mode == ModeKeys.TRAIN:
            return self.model_train_fn(mode, features, params)
        elif mode == ModeKeys.PREDICT:
            return self.model_gen_samples_fn(mode, features, params)

    def rnet(self, images, is_training=False, scope='rnet'):
        with tf.name_scope(scope):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                normalizer_fn=slim.batch_norm,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='valid'):
                with slim.arg_scope([slim.batch_norm], is_training = is_training):
                    net = slim.conv2d(images, num_outputs=28, kernel_size=[3, 3], stride=1)
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding='SAME')
                    net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1)
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
                    net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1)
                    fc_flatten = slim.flatten(net)
                    fc1 = slim.fully_connected(fc_flatten, num_outputs=128)
                    pred_logit = slim.fully_connected(fc1, num_outputs=2, activation_fn=None, normalizer_fn=None)
                    pred_cls = slim.softmax(pred_logit)[...,1]
                    pred_loc = slim.fully_connected(fc1, num_outputs=4, activation_fn=None, normalizer_fn=None)
                    return {'pred_logit': pred_logit,
                            'pred_cls': pred_cls,
                            'pred_loc': pred_loc}

    def model_train_fn(self, mode, features, params):
        images = features['image']
        gt_cls, gt_loc = features['gt_cls'], features['gt_loc']

        predictions = self.rnet(images, is_training=True)
        pred_cls, pred_loc = predictions['pred_cls'], predictions['pred_loc']
        thres = params['thres']

        optimizer = tf.train.AdamOptimizer(learning_rate=tfe.lr_plateau_decay(decay=0.9995))
        step = tf.train.get_or_create_global_step()
        loss = self.loss_fn(gt_cls, gt_loc, pred_cls, pred_loc, match_thres=thres)
        regular_loss = tf.add_n(tf.losses.get_regularization_losses())
        total_loss = loss['total_loss'] + regular_loss
        minimize_op = optimizer.minimize(total_loss, step)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_op)

        #summary
        self.metrics_fn(gt_cls, pred_cls, match_thres=thres)
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            tf.summary.histogram(var.name, var)
        return tf.estimator.EstimatorSpec(mode, predictions, total_loss, train_op)

    def model_gen_samples_fn(self, mode, features, params):
        assert mode == ModeKeys.PREDICT
        sample_imgs = features['sample_imgs']
        sample_bboxes = features['sample_bboxes']
        sample_imgs, sample_bboxes= tf.cond(pred=tf.equal(tf.size(sample_bboxes), 0),
                         true_fn=lambda: (tf.zeros([1,24,24,3]), tf.zeros([1,4])),
                         false_fn=lambda: (sample_imgs,sample_bboxes))
        thres = params['thres']

        predictions = self.rnet(sample_imgs, is_training=False)
        pred_cls, pred_loc = predictions['pred_cls'], predictions['pred_loc']
        pred_bboxes, pred_scores = tfe.decode_pred_cls_loc(pred_cls, pred_loc, sample_bboxes, thres)
        nms_idx = tf.image.non_max_suppression(pred_bboxes, pred_scores, iou_threshold=0.5,
                                               max_output_size=200, score_threshold=thres)
        nms_scores = tf.expand_dims(tf.gather(pred_scores, nms_idx), 0)
        nms_bboxes = tf.expand_dims(tf.gather(pred_bboxes, nms_idx), 0)
        predictions = {'bboxes':nms_bboxes, 'scores':nms_scores}
        hooks = [tf.train.LoggingTensorHook([], every_n_iter=100)]
        return tf.estimator.EstimatorSpec(mode, predictions, prediction_hooks=hooks)


    def loss_fn(self, gt_cls, gt_loc, pred_cls, pred_loc, match_thres):
        with tf.name_scope('loss_fn'):
            pmask = gt_cls > match_thres
            nmask = tf.logical_not(pmask)
            gt_binary = tf.cast(pmask, tf.float32)
            num_pos = tf.reduce_sum(gt_binary)
            num_neg = tf.reduce_sum(1-gt_binary)
            gt_cls = tf.stop_gradient(gt_cls)
            gt_loc = tf.stop_gradient(gt_loc)

            pos_loss = tf.losses.mean_squared_error(tf.boolean_mask(pred_cls, pmask),
                                                    tf.boolean_mask(gt_cls, pmask))
            neg_loss = tf.losses.mean_squared_error(tf.boolean_mask(pred_cls, nmask),
                                                    tf.boolean_mask(gt_cls, nmask))
            loc_loss = tf.losses.mean_squared_error(tf.boolean_mask(pred_loc, pmask),
                                                    tf.boolean_mask(gt_loc, pmask))
            cls_loss = 2 * pos_loss + neg_loss
            total_loss = cls_loss + loc_loss

            #summary
            tf.summary.scalar('num_pos', num_pos)
            tf.summary.scalar('num_neg', num_neg)
            tf.summary.scalar('pos_loss', pos_loss)
            tf.summary.scalar('neg_loss', neg_loss)
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('loc_loss', loc_loss)
            tf.summary.scalar('total_loss', total_loss)
            return {'cls_loss': cls_loss,
                    'loc_loss': loc_loss,
                    'total_loss': total_loss}

    def metrics_fn(self,gt_cls, pred_cls, match_thres, scope='metrics_fn'):
        with tf.name_scope(scope):
            gt_cls = gt_cls > match_thres
            pred_cls = pred_cls > match_thres
            cls_acc = tfe.accuracy(gt_cls, pred_cls)
            cls_prec = tfe.precision(gt_cls, pred_cls)
            cls_recall = tfe.recall(gt_cls, pred_cls)
            for k, v in tfe.TP_FP_TN_FN(gt_cls, pred_cls).items():
                tf.summary.scalar(k,v)
            tf.summary.scalar('cls_acc', cls_acc)
            tf.summary.scalar('cls_recall', cls_recall)
            tf.summary.scalar('cls_prec', cls_prec)
        return {'acc': cls_acc,
                'pred': cls_prec,
                'recall': cls_recall}

    def bboxes_metrics_fn(self, gt_bboxes, pred_bboxes, thres, scope='bboxes_metrics_fn'):
        with tf.name_scope(scope):
            prec = tfe.bboxes_precision(gt_bboxes, pred_bboxes, thres)
            recall = tfe.bboxes_recall(gt_bboxes, pred_bboxes, thres)
            return prec, recall
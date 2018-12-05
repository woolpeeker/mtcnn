
import tensorflow as tf
import numpy as np

import tf_extend as tfe

class Anchor:
    def __init__(self, anchor_shape, anchor_step=10 ,anchor_size=20, align_corner=False):
        #aligner_corner make anchor's corner is aligned with image corner
        self.anchor_shape = anchor_shape
        self.loc_shape = tf.concat([self.anchor_shape, (4,)], axis = 0)
        self.anchor_size = anchor_size
        self.anchor_step = anchor_step
        self.align_corner = align_corner
        yxyx, yxhw = self.get_anchors(anchor_shape, anchor_size, anchor_step, align_corner)
        self.yxyx = yxyx
        self.yxhw = yxhw
        self.ruler = self.anchor_size

    def get_anchors(self, anchor_shape, anchor_size, anchor_step, align_corner):
        x, y = tf.meshgrid(tf.range(anchor_shape[1]),
                           tf.range(anchor_shape[0]))
        x, y = y*anchor_step, x*anchor_step
        x, y = tf.cast(y, tf.float32), tf.cast(x, tf.float32)
        if align_corner:
            ymin, ymax = y, y + anchor_size
            xmin, xmax = x, x + anchor_size
        else:
            ymin, ymax = y - anchor_size / 2, y + anchor_size / 2
            xmin, xmax = x - anchor_size / 2, x + anchor_size / 2
        anchors = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        anchors = tf.cast(anchors, tf.float32)
        anchors_yxhw = tfe.yxyx2yxhw(anchors)
        return anchors, anchors_yxhw

    def bboxes_encode(self, bboxes, scope='bboxes_encode'):
        """Encode labels and bounding boxes."""
        with tf.name_scope(scope):
            anchors = tf.reshape(self.yxyx, shape=[-1, 4])
            bboxes = tf.cast(bboxes, tf.float32)
            def get_cls(bbox):
                return tfe.bboxes_jaccard(bbox, anchors)
            def get_loc(bbox):
                bbox = tfe.yxyx2yxhw(bbox)
                yxhw = tf.reshape(self.yxhw, [-1,4])
                loc=(bbox-yxhw)/self.ruler
                return loc

            def has_bboxes():
                cls = tf.map_fn(fn=get_cls,
                                elems=bboxes,
                                dtype=tf.float32,
                                back_prop=False)
                loc = tf.map_fn(fn=get_loc,
                                elems=bboxes,
                                dtype=tf.float32,
                                back_prop=False)
                idx = tf.cast(tf.argmax(cls, axis=0), tf.int32)
                idx = tf.stack([idx, tf.range(tf.shape(cls)[-1])], axis=-1)
                loc = tf.gather_nd(loc, idx)
                loc = tf.reshape(loc, self.loc_shape)
                cls = tf.reduce_max(cls, axis=0)
                cls = tf.reshape(cls, self.anchor_shape)
                return cls, loc

            def has_no_bboxes():
                zero_cls = tf.zeros(self.anchor_shape, tf.float32)
                zero_loc = tf.zeros(self.loc_shape, tf.float32)
                return zero_cls, zero_loc

            return tf.cond(pred=tf.shape(bboxes)[0] > 0,
                           true_fn=has_bboxes,
                           false_fn=has_no_bboxes)

    def bboxes_decode(self, cls, loc, match_thres, scope='bboxes_decode'):
        """Decode labels and bounding boxes."""
        with tf.name_scope(scope):
            assert len(tfe.get_shape(cls)) == 3
            keep = tf.minimum(tf.size(cls),25)
            anchors = tf.reshape(self.yxhw, [-1, 4])

            def fn(args):
                cls, loc = args
                fcls = tf.reshape(cls, [-1])
                floc = tf.reshape(loc, [-1, 4])
                v, idx = tf.nn.top_k(fcls, keep, sorted=True)
                sel_anchors = tf.gather(anchors, idx)
                floc = tf.gather(floc, idx)
                bboxes = sel_anchors + floc*self.ruler
                bboxes = tfe.yxhw2yxyx(bboxes)
                bboxes = tf.where(v > match_thres,
                                  bboxes,
                                  tf.zeros_like(bboxes))
                v = tf.where(v > match_thres,
                                  v,
                                  tf.zeros_like(v))
                return bboxes, v

            bboxes, scores = tf.map_fn(fn=fn,
                               elems=[cls, loc],
                               back_prop=False,
                               dtype=(tf.float32, tf.float32))
            return bboxes, scores


def encode_pred_bboxes(pred_bboxes, gt_bboxes, scope='encode_pred_bboxes'):
    # pred_bboxes and gt_bboxes should be [num, 4], [num, 4]
    with tf.name_scope(scope):
        # check if pred_bboxes is empty
        def has_no_bboxes():
            cls = tf.zeros(tf.shape(pred_bboxes)[0], dtype=tf.float32)
            loc = tf.zeros_like(pred_bboxes,dtype=tf.float32)
            return cls, loc
        def has_bboxes(pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes):
            gt_yxyx = tf.reshape(gt_bboxes, [-1,4])
            gt_yxhw = tfe.yxyx2yxhw(gt_yxyx)
            def get_cls(bbox):
                return tfe.bboxes_jaccard(bbox, gt_bboxes)
            def get_loc(bbox):
                bbox = tfe.yxyx2yxhw(bbox)
                ruler = tf.concat([bbox[2:4], bbox[2:4]], axis=-1)
                loc = (bbox - gt_yxhw) / ruler
                return loc
            cls = tf.map_fn(fn=get_cls,
                            elems=pred_bboxes,
                            dtype=tf.float32,
                            back_prop=False)
            loc = tf.map_fn(fn=get_loc,
                            elems=pred_bboxes,
                            dtype=tf.float32,
                            back_prop=False)
            idx = tf.cast(tf.argmax(cls, axis=-1), tf.int32)
            idx = tf.stack([tf.range(tf.shape(idx)[0]), idx], axis=-1)
            loc = tf.gather_nd(loc, idx)
            cls = tf.reduce_max(cls, axis=-1)
            return cls, loc

        condition = tf.logical_or(tf.equal(tf.size(gt_bboxes),0),
                                  tf.equal(tf.size(pred_bboxes),0))
        cls, loc = tf.cond(pred = condition,
                true_fn = has_no_bboxes,
                false_fn = has_bboxes)
        return cls, loc

def decode_pred_cls_loc(pred_cls, pred_loc, bboxes, thres, scope='decode_pred_cls_loc'):
    #bboxes is the output bboxes of pnet or rnet
    with tf.name_scope(scope):
        bboxes = tfe.yxyx2yxhw(bboxes)
        ruler = tf.concat([bboxes[...,2:4], bboxes[...,2:4]], -1)
        decoded_bboxes = bboxes - pred_loc*ruler
        decoded_bboxes = tfe.yxhw2yxyx(decoded_bboxes)

        mask = pred_cls > thres
        scores = tf.boolean_mask(pred_cls, mask)
        decoded_bboxes = tf.boolean_mask(decoded_bboxes, mask)
        return decoded_bboxes, scores

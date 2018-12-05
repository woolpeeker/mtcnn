# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-processing images for SSD-type networks.
"""
from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf
import tf_extend as tfe

from tensorflow.python.ops import control_flow_ops


slim = tf.contrib.slim

BBOX_CROP_OVERLAP = 0.5
MIN_OBJECT_COVERED = 0.5
CROP_RATIO_RANGE = (0.95, 1.05)
AREA_RANGE = (0.01, 1)

def preprocess_random_scale(image, bboxes, out_shape, isBboxInt=True,
                            scope='preprocess_random_scale'):
    with tf.name_scope(scope):
        out_shape = tf.to_int32(out_shape)
        image_shape = tfe.img_shape(image, tf.float32)
        if isBboxInt:
            bboxes = tfe.convert_bboxes_to_float(bboxes, image_shape)
        short_side = tf.reduce_min(image_shape)
        scale = tf.random_uniform([], minval=tf.to_float(out_shape[0])/short_side, maxval=1.)
        target_shape = tf.to_int32(scale * image_shape)
        target_shape = tf.maximum(target_shape, out_shape+1)
        resized_image = tfe.resize_image_no_batch(image, target_shape)
        dst_y = tf.random_uniform([], minval=0, maxval=target_shape[0]-out_shape[0], dtype=tf.int32)
        dst_x = tf.random_uniform([], minval=0, maxval=target_shape[1]-out_shape[1], dtype=tf.int32)
        dst_h = out_shape[0]
        dst_w = out_shape[1]
        dst_bbox = tf.stack([dst_y/target_shape[0],
                             dst_x/target_shape[1],
                             (dst_y + dst_h)/target_shape[0],
                             (dst_x+dst_w)/target_shape[1]])
        dst_image = tf.image.crop_to_bounding_box(resized_image, dst_y, dst_x, dst_h, dst_w)
        bboxes = tfe.bboxes_resize(tf.to_float(dst_bbox), bboxes)
        _, bboxes = tfe.bboxes_filter_center(tf.ones([tf.shape(bboxes)[0]]), bboxes)
        bboxes = tfe.convert_bboxes_to_int(bboxes, out_shape)
        mask = tf.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 5, (bboxes[:, 3] - bboxes[:, 1]) > 5)
        bboxes = tf.boolean_mask(bboxes, mask)
        dst_image = tf.cast(dst_image, tf.float32)
        return {'image':dst_image, 'bboxes':bboxes}


def preprocess_choose_single_bbox(image, bboxes, scale_bbox_to, out_shape, isBboxInt=True,
                                  scope='preprocess_choose_single_bbox'):
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        im_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                element_shape=[out_shape[0], out_shape[1],3])
        bb_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                    element_shape=[None,4])
        def body(i, im_arr, bb_arr):
            sel_bbox = bboxes[i]
            dst_im, dst_bb = preprocess_choose_single_bbox_worker(image, bboxes, sel_bbox,
                                                                  scale_bbox_to=scale_bbox_to, out_shape=out_shape,
                                                                  isBboxInt=isBboxInt)
            i+=1
            im_arr = im_arr.write(i, dst_im)
            bb_arr = bb_arr.write(i, dst_bb)
            return i, im_arr, bb_arr
        _, im_arr, bb_arr = tf.while_loop(loop_vars=[0, im_arr, bb_arr],
                                          cond=lambda i, _i, _b: i<tf.shape(bboxes)[0],
                                          body=body,
                                          parallel_iterations=20, back_prop=False)
        im_all = im_arr.stack()
        bb_all = bb_arr.stack()
        return tf.data.Dataset.from_tensor_slices({'image':im_all, 'bboxes': bb_all})

def preprocess_choose_single_bbox_worker(image, bboxes, sel_bbox, scale_bbox_to, out_shape, isBboxInt=True, keep = 100,
                                  scope='proprocess_choose_single_bbox_worker'):
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        image_shape = tfe.img_shape(image, tf.float32)
        scale_bbox_to = tf.to_float(scale_bbox_to)
        if isBboxInt:
            bboxes = tfe.convert_bboxes_to_float(bboxes, image_shape)
        scale = scale_bbox_to / tf.sqrt((sel_bbox[2] - sel_bbox[0]) * (sel_bbox[3] - sel_bbox[1]))
        sel_bbox = tfe.convert_bboxes_to_float(sel_bbox, image_shape)
        scale = tf.random_uniform([2], scale * 0.9, scale * 1.1)
        scale = scale * tf.random_uniform([], 0.6, 1.4)
        scale = tf.cond(pred=tf.logical_or(tf.is_nan(scale[0]), tf.is_nan(scale[1])),
                        true_fn=lambda: tf.ones([2]),
                        false_fn=lambda: scale)
        image_shape = tf.to_int32(image_shape * scale)
        image_shape = tf.maximum(image_shape, out_shape)
        resized_image = tfe.resize_image_no_batch(image, image_shape)
        image_shape = tf.to_float(image_shape)

        sel_bbox = tf.clip_by_value(sel_bbox, 0., 1.)
        dst_h = tf.to_float(out_shape[0] / image_shape[0])
        dst_w = tf.to_float(out_shape[1] / image_shape[1])
        dst_y = tf.random_uniform([], minval=tf.maximum(0., sel_bbox[2] - dst_h),
                                  maxval=tf.minimum(sel_bbox[0], 1. - dst_h))
        dst_x = tf.random_uniform([], minval=tf.maximum(0., sel_bbox[3] - dst_w),
                                  maxval=tf.minimum(sel_bbox[1], 1. - dst_w))
        dst_bbox = tf.stack([dst_y, dst_x, dst_y + dst_h, dst_x + dst_w])
        dst_image = tf.image.crop_to_bounding_box(resized_image,
                                                  tf.to_int32(image_shape[0] * dst_y),
                                                  tf.to_int32(image_shape[1] * dst_x),
                                                  out_shape[0],
                                                  out_shape[1])
        bboxes = tfe.bboxes_resize(dst_bbox, bboxes)
        _, bboxes = tfe.bboxes_filter_center(tf.ones([tf.shape(bboxes)[0]]), bboxes)
        bboxes = tfe.convert_bboxes_to_int(bboxes, out_shape)
        mask = tf.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 6, (bboxes[:, 3] - bboxes[:, 1]) > 6)
        bboxes = tf.boolean_mask(bboxes, mask)
        bboxes = tfe.pad_axis(bboxes, 0, keep, axis=0)
        dst_image = tf.cast(dst_image, tf.float32)
        #dst_image = tf.cond(pred=tf.reduce_all(dst_image<1e-6),
        #                      true_fn=lambda: tf.Print(dst_image,[dst_bbox, image_shape, scale], 'all_zeros: dst_bbox, image_shape, scale', summarize=50),
        #                      false_fn=lambda: dst_image)
        return dst_image, bboxes


def preprocess_for_train_pnet(image, bboxes, labels, out_shape, isBboxInt=True,
                         scope='preprocess_for_train_pnet'):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.
    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope, 'preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Distort image and bounding boxes.
        image_shape = tfe.img_shape(image)
        if isBboxInt:
            bboxes = tfe.convert_bboxes_to_float(bboxes,image_shape)
        bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)
        #resize
        mean_size_box = tf.cond(pred=tf.equal(tf.shape(bboxes)[0], 0),
                                true_fn=lambda: tf.constant(20 / 60.0, tf.float32),
                                false_fn=lambda: tf.reduce_mean(bboxes[:, 2] - bboxes[:, 0]))
        scale = 20.0 / 60.0 / mean_size_box
        scale = tf.minimum(tf.maximum(0.1, scale), 1)
        scale = tf.random_shuffle(tf.stack([1.0, scale], axis=0))[0]
        target_shape = scale * tf.cast(image_shape, tf.float32)
        target_shape = tf.cast(target_shape, tf.int32)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bilinear(image, target_shape, align_corners=True)
        image = tf.image.resize_image_with_crop_or_pad(image, image_shape[0], image_shape[1])
        image = tf.squeeze(image, axis=0)
        bboxes = bboxes * scale + (1 - scale) / 2

        #crop
        bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)
        dst_image, labels, bboxes, distort_bbox = \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        min_object_covered=MIN_OBJECT_COVERED,
                                        aspect_ratio_range=CROP_RATIO_RANGE,
                                        area_range=AREA_RANGE)
        # Resize image to output size.
        dst_image = tfe.resize_image_no_batch(dst_image, out_shape)
        bboxes = tfe.convert_bboxes_to_int(bboxes, tfe.img_shape(dst_image))
        return {"image": dst_image, 'bboxes': bboxes, "labels": labels}

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
    return cropped_image, labels, bboxes, distort_bbox

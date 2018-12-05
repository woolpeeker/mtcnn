
import numpy as np

def convert_bboxes_to_float(bboxes, image_shape):
    bboxes = [bboxes[..., 0] / image_shape[0],
              bboxes[..., 1] / image_shape[1],
              bboxes[..., 2] / image_shape[0],
              bboxes[..., 3] / image_shape[1]]
    bboxes = np.stack(bboxes,axis=-1)
    return bboxes

def convert_bboxes_to_int(bboxes, image_shape):
    bboxes = [bboxes[..., 0] * image_shape[0],
              bboxes[..., 1] * image_shape[1],
              bboxes[..., 2] * image_shape[0],
              bboxes[..., 3] * image_shape[1]]
    bboxes = np.stack(bboxes, axis=-1)
    return bboxes

def bboxes_filter_center(labels, bboxes, margins=(0., 0., 0., 0.)):
    """Filter out bounding boxes whose center are not in
    the rectangle [0, 0, 1, 1] + margins. The margin Tensor
    can be used to enforce or loosen this condition.

    Return:
      labels, bboxes: Filtered elements.
    """
    cy = (bboxes[:, 0] + bboxes[:, 2]) / 2.
    cx = (bboxes[:, 1] + bboxes[:, 3]) / 2.
    mask = cy > margins[0]
    mask = np.logical_and(mask, cx > margins[1])
    mask = np.logical_and(mask, cy < 1 + margins[2])
    mask = np.logical_and(mask, cx < 1 + margins[3])
    # Boolean masking...
    labels = labels[mask]
    bboxes = bboxes[mask]
    return labels, bboxes

def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    v = np.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
    bboxes = bboxes - v
    # Scale.
    s = np.stack([bbox_ref[2] - bbox_ref[0],
                  bbox_ref[3] - bbox_ref[1],
                  bbox_ref[2] - bbox_ref[0],
                  bbox_ref[3] - bbox_ref[1]])
    bboxes = bboxes / s
    return bboxes
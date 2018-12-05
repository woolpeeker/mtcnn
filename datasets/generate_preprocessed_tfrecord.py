import numpy as np
import multiprocessing as mp
import PIL.Image as Image
import re, os, io
from math import ceil

import np_extend as npe
import tf_extend as tfe

widerPath = '/home/luojiapeng/datasets/widerface'
imagesDir = widerPath + '/WIDER_train/images'
annoFile = widerPath + '/wider_face_split/wider_face_train_bbx_gt.txt'
#annoFile = widerPath + '/wider_face_split/wider_face_demo_bbx_gt.txt'
outShape = (80,80)
scaleTo = 12
repeat = 50
outPath = 'wider_train_cropped%d_scaleTo%d_repeat%d.tfrecord'%(outShape[0], scaleTo, repeat)
#outPath = 'wider_demo_cropped_80_scaleTo_15.tfrecord'

def get_annotations(annoPath):
    annoFile = open(annoPath, 'r')
    annotations=[]
    for i,line in enumerate(annoFile.readlines()):
        line=line.rstrip()
        if re.match('^.*\.jpg$',line):
            annotations.append([line])
        elif len(line.split())>4:
            info=[int(x) for x in line.split()[:4]]
            #change xywh to yxyx
            info=[info[1], info[0], info[3]+info[1], info[2]+info[0]]
            annotations[-1]=annotations[-1] + info
    return annotations

def process_single_anno(fname, bboxes):
    result=[]
    img = Image.open(fname)
    for bbox in bboxes:
        if not np.all(bbox[-2:]-bbox[:2]>8):
            continue
        cropped_img, sample_bboxes = crop_with_bboxes(img, bbox, bboxes)
        result.append({'image':cropped_img, 'bboxes':sample_bboxes})
    return result

def crop_with_bboxes(img, bbox, bboxes):
    imgShape=(img.size[1], img.size[0])
    scale = scaleTo / np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    scale = np.random.uniform(0.9*scale, 1.1*scale, size=[2])
    scale = scale * np.random.uniform(0.65, 1.35)
    resizedImgShape = np.round(scale * imgShape).astype(np.int32)
    resizedImgShape = np.maximum(resizedImgShape, outShape)
    resizedImg = img.resize((resizedImgShape[1], resizedImgShape[0]),
                            Image.BILINEAR)
    bbox = npe.convert_bboxes_to_float(bbox, imgShape)
    bboxes = npe.convert_bboxes_to_float(bboxes, imgShape)
    bbox = np.clip(bbox, 0, 1)
    bboxes = np.clip(bboxes, 0, 1)

    dst_h = outShape[0] / resizedImgShape[0]
    dst_w = outShape[1] / resizedImgShape[1]
    dst_y = np.random.uniform(low=np.maximum(0, bbox[2] - dst_h),
                              high=np.minimum(bbox[0], 1. - dst_h))
    dst_x = np.random.uniform(low=np.maximum(0, bbox[3] - dst_w),
                              high=np.minimum(bbox[1], 1. - dst_w))
    dst_bbox = np.stack([dst_y, dst_x, dst_y + dst_h, dst_x + dst_w])
    dst_bbox_int = np.round(npe.convert_bboxes_to_int(dst_bbox, resizedImgShape)).astype(np.int32)
    croppedImg = resizedImg.crop([dst_bbox_int[1],
                                  dst_bbox_int[0],
                                  dst_bbox_int[3],
                                  dst_bbox_int[2]])
    bboxes = npe.bboxes_resize(dst_bbox, bboxes)
    _, bboxes = npe.bboxes_filter_center(np.ones(len(bboxes)), bboxes)
    bboxes = np.round(npe.convert_bboxes_to_int(bboxes, outShape))
    mask = np.all(bboxes[:,[2,3]]-bboxes[:,[0,1]] > 5, axis=-1)
    bboxes = bboxes[mask]
    assert croppedImg.size==outShape
    imgByteArr = io.BytesIO()
    croppedImg.save(imgByteArr, format='JPEG')
    croppedImg = imgByteArr.getvalue()
    return croppedImg, bboxes

def work_fn(anno):
    fname = anno[0]
    bboxes = np.array(anno[1:]).reshape([-1, 4])
    samples = process_single_anno(os.path.join(imagesDir, fname), bboxes)
    return samples


def sample_generate_fn(annotations, repeat):
    pool = mp.Pool(processes=20)
    for i in range(repeat):
        step=2000
        for j in range(0, ceil(len(annotations)/step)):
            print('%d epochs, %d samples' % (i, j*step))
            annoChunk = annotations[j*step : min((j+1)*step,len(annotations))]
            samplesChunk = pool.map(work_fn, annoChunk, 50)
            for samples in samplesChunk:
                for sample in samples:
                    yield sample

if __name__ == '__main__':
    annotations = get_annotations(annoFile)
    sample_generator = sample_generate_fn(annotations, repeat)

    tfrecordObj = tfe.ImageBboxTfrecord()
    tfrecordObj.write(outPath, sample_generator)

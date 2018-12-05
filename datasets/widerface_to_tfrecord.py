
import PIL.Image as Image
import os, io

import numpy as np
import re
import tensorflow as tf

import multiprocessing as mp

import tf_extend as tfe


widerPath = '/home/luojiapeng/datasets/widerface'
imagesDir = widerPath + '/WIDER_train/images'
annoFile = widerPath + '/wider_face_split/wider_face_train_bbx_gt.txt'
outPath = 'wider_train.tfrecord'



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

def sample_generate_fn(annotations):
    count = 0
    for anno in annotations:
        count+=1
        if not count%100:
            print('count: %d'%count)
        fname = os.path.join(imagesDir, anno[0])
        bboxes = np.array(anno[1:]).reshape([-1, 4])
        image = Image.open(fname,'r')
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='JPEG')
        image = imgByteArr.getvalue()
        yield {'image':image,
               'bboxes':bboxes}


if __name__ == "__main__":
    annotations = get_annotations(annoFile)
    sample_generator = sample_generate_fn(annotations)

    tfrecordObj = tfe.ImageBboxTfrecord()
    tfrecordObj.write(outPath, sample_generator)

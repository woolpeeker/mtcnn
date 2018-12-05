
import tf_extend as tfe

def get_dataset(fname='datasets/wider_train.tfrecord'):
    tfrecordObj = tfe.ImageBboxTfrecord()
    dataset = tfrecordObj.read(fname,n_thread=16)
    return dataset
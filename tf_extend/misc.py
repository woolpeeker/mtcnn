import os, re
import tensorflow as tf

def get_checkpoint_dir(path='./checkpoints', fold='run', num=None):
    assert os.path.exists(path)
    if num is not None:
        return os.path.join(path,fold+'_'+str(num))
    max_run=-1
    for d in os.listdir(path):
        d=os.path.join(path,d)
        if os.path.isdir(d):
            match=re.search('[\\/]%s_(\d+)$'%fold,d)
            if match:
                num=int(match.group(1))
                max_run=num if num>max_run else max_run
    run_num=max_run+1
    result_path=os.path.join(path,fold+'_'+str(run_num))
    tf.logging.info('get_checkpoint_dir: %s'%result_path)
    return result_path

def lr_plateau_decay(lr=0.01,decay=0.999, min_lr=1e-5, loss=None, scope='lr_plateau_decay'):
    with tf.name_scope(scope):
        his_len = 10
        local_lr = tf.get_local_variable(name='local_lr', dtype=tf.float32,
                                         initializer=tf.constant(lr, dtype=tf.float32))
        loss_idx = tf.get_local_variable(name='loss_idx', dtype=tf.int32,
                                         initializer=tf.constant(1,dtype=tf.int32))
        his_loss = tf.get_local_variable(name='history_loss',dtype=tf.float32,
                                             initializer=tf.zeros([his_len])-1.0)
        if loss is None:
            loss = tf.losses.get_total_loss()
        update_history = tf.assign(his_loss[loss_idx], loss)
        with tf.control_dependencies([update_history]):
            update_idx = tf.assign(loss_idx, tf.mod(loss_idx + 1, his_len))
        with tf.control_dependencies([update_idx]):
            updated_lr = tf.cond(pred=loss>tf.reduce_mean(his_loss),
                                 true_fn=lambda: tf.assign(local_lr, local_lr*decay),
                                 false_fn=lambda: local_lr)
        lr = tf.maximum(updated_lr, min_lr)
        tf.summary.scalar('lr_plateau_decay', lr)
    return lr


import tensorflow as tf
import tf_extend as tfe

def fake_quantize(x, intBits=10, fracBits=10, isFloor=False, scope='fake_quantize'):
    with tf.name_scope(scope):
        x=tf.clip_by_value(x, -2**intBits, 2**intBits)
        if isFloor:
            func = tf.floor
        else:
            func = tf.round
        fq = func(x*(2**fracBits))/(2**fracBits)
        fq = tf.clip_by_value(fq, -2**intBits+1./2**fracBits, 2**intBits-1./2**fracBits)
        return x + tf.stop_gradient(fq - x)

class quantizeHook(tf.train.SessionRunHook):
    def __init__(self, quantize_fn=fake_quantize, print=True):
        self.hasQuantized = False
        self.quantize_fn = quantize_fn
        self.print = print

    def begin(self):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        quantizeOps = []
        printOps = []
        for var in vars:
            if 'global_step' in var.name:
                continue
            else:
                if self.print:
                    printOp = tf.print(var.name, ' min, max, mean, median',
                                       tf.reduce_min(var),
                                       tf.reduce_max(var),
                                       tf.reduce_mean(var),
                                       tfe.get_median(var))
                    with tf.control_dependencies([printOp]):
                        quantizeOp = tf.assign(var, self.quantize_fn(var))
                    quantizeOps.append(quantizeOp)
                    printOps.append(printOp)
        self.quantizeOps = quantizeOps
        self.quantizeOps.append(tf.print("QuantizeOps executed.", output_stream=tf.logging.info))
        self.printOps = printOps

    def before_run(self, run_context):
        if not self.hasQuantized:
            self.hasQuantized = True
            if self.print:
                Ops = [*self.printOps, *self.quantizeOps]
            else:
                Ops = self.quantizeOps
            return tf.train.SessionRunArgs(Ops)
        else:
            return None
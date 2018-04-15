import numpy as np
import tensorflow as tf

def get_trainable_variables(flist, filter_type="in"):
    all_vs = tf.trainable_variables()
    if filter_type=="in":
        vs = []
        for s in flist:
            vs += [p for p in all_vs if s in p.name]
    elif filter_type=="not in":
        vs = all_vs
        for s in flist:
            vs = [p for p in vs if s not in p.name]

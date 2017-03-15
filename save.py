#!/usr/bin/env python2
import cv2
import os, sys
from os import path
from os.path import join

import numpy as np

from keras.models import load_model
from video import load_vid

from keras import backend as K

def save_output(input_dir, output_dir, model):
    i = 0
    total = 0
    for root, dirs, files in os.walk(input_dir):
        infiles = filter(lambda x: x.lower().endswith('.avi') or x.lower().endswith('.npy'), files)
        total += len(infiles)

    for root, dirs, files in os.walk(input_dir):
        infiles = filter(lambda x: x.lower().endswith('.avi') or x.lower().endswith('.npy'), files)
        for infile in infiles:
            i += 1
            out_file = join(root.replace(input_dir, output_dir, 1), infile)
            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if infile.lower().endswith('.npy'): # optical flows
                flow_x = np.load(join(root,infile)).astype('float32')
                flow_x *= 1./255
                nb_flows = flow_x.shape[2] // 2 - 9
                output = np.zeros((nb_flows,flow_x.shape[0],flow_x.shape[1],20), dtype='float32')
                for k in range(nb_flows):
                    output[k] = flow_x[:,:,k*2:k*2+20]
                output = model.predict(output)

            else:   # video
                out_file += '.npy'  # file extension
                output = load_vid(join(root,infile), dim_ordering='tf')
                output *= 1./255
                output = model.predict(output)

            print "(%d/%d) Saving to %s" % (i, total, out_file),
            print output.shape
            np.save(out_file, output)

if len(sys.argv) < 4:
    print "Usage: %s [input_dir] [output_dir] [model_path] [split=-1(1,2 or 3. -1 when don't use split number)] [set_type=train, val or both. works iff split > -1]" % sys.argv[0]
    print "Author: Kiyoon Kim (yoonkr33@gmail.com)"
    print "predict video, numpy(optical flow) data with a model and save outputs"
    sys.exit()

gpu_frac = 0.3
# set GPU memory usage fraction
if K.backend() == 'tensorflow':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    set_session(tf.Session(config=config))


input_dir = sys.argv[1]
output_dir = sys.argv[2]
model_path = sys.argv[3]


dataset_split = -1
if len(sys.argv) >= 5:
    dataset_split = int(sys.argv[4])

set_type = 'both'
if len(sys.argv) >= 6:
    set_type = sys.argv[5]

model = load_model(model_path)
model.pop()
model.pop()
model.pop()
model.compile(optimizer='sgd', loss='categorical_crossentropy')

if input_dir.endswith('/'):
    input_dir = input_dir[:-1]
if output_dir.endswith('/'):
    output_dir = output_dir[:-1]

if dataset_split >= 0:
    if set_type == 'both' or set_type == 'train':
        save_output(join(input_dir, 'train%d' % dataset_split), join(output_dir, 'train%d' % dataset_split), model)
    if set_type == 'both' or set_type == 'val':
        save_output(join(input_dir, 'val%d' % dataset_split), join(output_dir, 'val%d' % dataset_split), model)
else:
    save_output(input_dir, output_dir, model)

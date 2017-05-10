"""Fairly basic set of tools for real-time data augmentation on video data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings

#from .. import backend as K
from keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift, transform_matrix_offset_center, apply_transform, flip_axis

import cv2  # version 2, 3 compatible


def load_vid(path, dim_ordering='default', grayscale=False, target_size=None):
    """Loads an video into numpy array format.

    # Arguments
        path: Path to video file
        dim_ordering: 'default', 'tf' or 'th'
        grayscale: Boolean, whether to load the video as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(frame_count, vid_height, vid_width)`.

    # Returns
        numpy array

    # Raises
        invalid dimension ordering value
    """

    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    cap = cv2.VideoCapture(path)
    if target_size is None:
        frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT if cv2.__version__[0] == '2' else cv2.CAP_PROP_FRAME_COUNT) + K.epsilon())
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__[0] == '2' else cv2.CAP_PROP_FRAME_HEIGHT) + K.epsilon())
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__[0] == '2' else cv2.CAP_PROP_FRAME_WIDTH) + K.epsilon())
        mode = 'same'
    else:
        frame_count = target_size[0]
        height = target_size[1]
        width = target_size[2]
        org_frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT if cv2.__version__[0] == '2' else cv2.CAP_PROP_FRAME_COUNT) + K.epsilon())
        if org_frame_count > frame_count:
            mode = 'omit'
            frame_ratio = float(org_frame_count)/frame_count - 1
        elif org_frame_count < frame_count:
            mode = 'duplicate'
            frame_ratio = float(frame_count)/org_frame_count - 1
        else:
            mode = 'same'

    if grayscale:
        channels = 1
    else:
        channels = 3

    if dim_ordering == 'tf':
        vid = np.zeros([frame_count,height,width,channels])
    else:
        vid = np.zeros([frame_count,channels,height,width])

    i = 0
    frame_ratio_sum = 0.0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if height != frame.shape[0] or width != frame.shape[1]:
            frame = cv2.resize(frame, (width, height))
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.reshape(frame, (height, width, 1))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if dim_ordering == 'th':
            frame = frame.transpose(2,0,1)

        vid[i] = frame
        i += 1
        if mode != 'same':
            frame_ratio_sum += frame_ratio
            while frame_ratio_sum > 1 - K.epsilon():
                frame_ratio_sum -= 1
                if mode == 'omit':
                    cap.read()
                else: # mode == 'duplicate'
                    vid[i] = vid[i-1]
                    i+=1
    if i != frame_count:
        raise Exception('Expected frame_count and real one doesn\'t match. Expected: ' + str(frame_count) + ' and real: ' + str(i)
                + '\nIt should not happen unless video.py itself is wrong.')

    vid = vid.astype('float32')
    cap.release()
    return vid

def save_vid(vid, path, rescale=None, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    if rescale is not None:
        vid = vid + max(-np.min(vid), 0)
        vid *= rescale
    vid = vid.astype('uint8') 

    if dim_ordering == 'tf':
        height = vid.shape[1]
        width = vid.shape[2]
        channel = vid.shape[3]
    else:
        height = vid.shape[2]
        width = vid.shape[3]
        channel = vid.shape[1]

    if cv2.__version__[0] == '2':
        fourcc = cv2.cv.CV_FOURCC(*'HFYU')  # Huffman lossless
    else:
        fourcc = cv2.FOURCC(*'HFYU')
    out = cv2.VideoWriter(path, fourcc, 25.0, (width,height))

    for frame in vid:
        if channel == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if dim_ordering == 'th':
            frame = frame.transpose(1,2,0)
        out.write(frame)

    out.release()

def list_videos(directory, ext='avi|mp4'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]


class Iterator(object):

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, x, y, video_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='avi'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (videos tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.x = np.asarray(x)
        if self.x.ndim != 5:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 4 if dim_ordering == 'tf' else 2
        if self.x.shape[channels_axis] not in {1, 3}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1 or 3 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.video_data_generator = video_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype='float32')
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.video_data_generator.random_transform(x.astype('float32'))
            x = self.video_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                save_vid(batch_x[i], os.path.join(self.save_to_dir, fname), rescale=1/self.video_data_generator.rescale if self.video_data_generator.rescale is not None else None)
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y



class VideoDataGenerator(object):
    """Generate minibatches of video data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one video (Numpy tensor with rank 4),
            and should output a Numpy tensor with the same shape.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.time_axis = 1
            self.channel_axis = 2
            self.row_axis = 3
            self.col_axis = 4
        if dim_ordering == 'tf':
            self.time_axis = 1
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='avi'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            target_size=(30, 256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='avi',
                            follow_links=False):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single video, so it doesn't have image number at index 0
        #vid_channel_axis = self.channel_axis - 1
        img_channel_axis = self.channel_axis - 2
        if self.samplewise_center:
            for i, xf in enumerate(x):    # frame by frame
                x[i] -= np.mean(xf, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            for i, xf in enumerate(x):
                x[i] /= (np.std(xf, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This VideoDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This VideoDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                for i, xf in enumerate(x):
                    flatx = np.reshape(xf, (xf.size))
                    whitex = np.dot(flatx, self.principal_components)
                    x[i] = np.reshape(whitex, (xf.shape[0], xf.shape[1], xf.shape[2]))
            else:
                warnings.warn('This VideoDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x):
        # x is a single video, so it doesn't have image number at index 0
        vid_time_axis = self.time_axis - 1
        vid_row_axis = self.row_axis - 1
        vid_col_axis = self.col_axis - 1
        vid_channel_axis = self.channel_axis - 1

        img_channel_axis = vid_channel_axis - 1
        img_row_axis = vid_row_axis - 1
        img_col_axis = vid_col_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[vid_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[vid_col_axis]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix),
                                  zoom_matrix)

        h, w = x.shape[vid_row_axis], x.shape[vid_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        for i, xf in enumerate(x):
            x[i] = apply_transform(xf, transform_matrix, img_channel_axis,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            for i, xf in enumerate(x):
                x[i] = random_channel_shift(xf,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                for i, xf in enumerate(x):
                    x[i] = flip_axis(xf, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                for i, xf in enumerate(x):
                    x[i] = flip_axis(xf, img_row_axis)

        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the video data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x)
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.dim_ordering + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1 or 3 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        # !! not tested
        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]))
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        # !! not tested
        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        # !! not tested
        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        # !! not tested
        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)

class DirectoryIterator(Iterator):

    def __init__(self, directory, video_data_generator,
                 target_size=(30, 256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='avi',
                 follow_links=False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.video_data_generator = video_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.video_shape = self.target_size + (3,)
            else:
                self.video_shape = list(self.target_size)
                self.video_shape.insert(1,3)
                self.video_shape = tuple(self.video_shape)
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.video_shape = list(self.target_size)
                self.video_shape.insert(1,1)
                self.video_shape = tuple(self.video_shape)
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'avi', 'mp4'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d videos belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.video_shape, dtype='float32')
        grayscale = self.color_mode == 'grayscale'
        # build batch of video data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_vid(os.path.join(self.directory, fname),
                           dim_ordering=self.dim_ordering,
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = self.video_data_generator.random_transform(x)
            x = self.video_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                save_vid(batch_x[i], os.path.join(self.save_to_dir, fname), rescale=1/self.video_data_generator.rescale if self.video_data_generator.rescale is not None else None)
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

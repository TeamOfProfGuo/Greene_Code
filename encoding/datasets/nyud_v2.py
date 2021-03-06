import os
import scipy.io
import pickle
import numpy as np
from PIL import Image

from .base import BaseDataset


class NYUD(BaseDataset):
    classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
            'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
            'ceiling','books','refridgerator','television','paper','towel','shower curtain','box','whiteboard','person',
            'night stand','toilet','sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

    NUM_CLASS = 40
    CWD = os.getcwd()
    if os.path.dirname(CWD).endswith('experiments'):
        BASE_DIR = '../../../dataset/NYUD_v2'
    else:
        BASE_DIR = '../dataset/NYUD_v2/'

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, dep_transform=None, target_transform=None, **kwargs):
        self.dep_transform = dep_transform
        print('==check dep_transform {}'.format(dep_transform))
        super(NYUD, self).__init__(root, split, mode, transform, target_transform, **kwargs)

        # train/val/test splits are pre-cut
        _nyu_root = os.path.abspath(self.BASE_DIR)
        _mask_dir = os.path.join(_nyu_root, 'nyu_labels40')
        _image_dir = os.path.join(_nyu_root, 'nyu_images')
        _depth_dir = os.path.join(_nyu_root, 'nyu_depths')
        if self.mode == 'train':
            _split_f = os.path.join(_nyu_root, 'splits/train.txt')
        else:
            _split_f = os.path.join(_nyu_root, 'splits/test.txt')
        self.images = []  # list of file names
        self.depths = []  # list of depth image
        self.masks = []   # list of file names
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                line = int(line.rstrip('\n')) - 1
                _image = os.path.join(_image_dir, str(line) + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _depth = os.path.join(_depth_dir, str(line) + '.png')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)
                _mask = os.path.join(_mask_dir, str(line) + ".png")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _dep = Image.open(self.depths[index])  # depth image with shape [h ,w]
        if self.mode == 'test':   # return image(tensor), depth(tensor) and (fileName)
            if self.transform is not None:
                _img = self.transform(_img)
            if self.dep_transform is not None:
                _dep = self.dep_transform(_dep)
            return _img, _dep, os.path.basename(self.images[index])

        _target = Image.open(self.masks[index])  # image with shape [h, w]
        # synchrosized transform
        if self.mode == 'train':
            # return _img (Image), _dep (Image), _target (2D tensor)
            _img, _dep, _target = self._sync_transform(_img, _target, depth=_dep, IGNORE_LABEL=0)
        elif self.mode == 'val':
            _img, _dep, _target = self._val_sync_transform(_img, _target, depth = _dep)

        _target -= 1  # since 0 represent the boundary
        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)  #_img to tensor, normalize
        if self.dep_transform is not None:
            _dep = self.dep_transform(_dep)  # depth to tensor, normalize
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _dep, _target  # all tensors

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

    def make_pred(self, x):
        return x

    def compute_class_weights(self, weight_mode='median_frequency', c=1.02):
        assert weight_mode in ['median_frequency', 'logarithmic', 'linear']

        # build filename
        class_weighting_filepath = os.path.join(self.BASE_DIR, 'weight', '{}.pickle'.format(weight_mode))

        if os.path.exists(class_weighting_filepath):
            class_weighting = pickle.load(open(class_weighting_filepath, 'rb'))
            print(f'Using {class_weighting_filepath} as class weighting')
            return class_weighting

        print('Compute class weights')

        LabelFile = os.path.join(self.BASE_DIR, 'nyuv2_meta/labels40.mat')
        data = scipy.io.loadmat(LabelFile)
        labels = np.array(data["labels40"])

        n_pixels_per_class = np.zeros(self.NUM_CLASS + 1)
        n_image_pixels_with_class = np.zeros(self.NUM_CLASS + 1)
        for i in range(1449):
            label = np.array(labels[:, :, i])
            h, w = label.shape
            current_dist = np.bincount(label.flatten(), minlength=self.NUM_CLASS + 1)
            n_pixels_per_class += current_dist

            # For median frequency we need the pixel sum of the images where the specific class is present.
            # (It only matters if the class is present in the image and not how many pixels it occupies.)
            class_in_image = current_dist > 0
            n_image_pixels_with_class += class_in_image * h * w
            print(f'\r{i + 1}/{len(self)}', end='')
        print()

        # remove void
        n_pixels_per_class = n_pixels_per_class[1:]
        n_image_pixels_with_class = n_image_pixels_with_class[1:]

        if weight_mode == 'linear':
            class_weighting = n_pixels_per_class

        elif weight_mode == 'median_frequency':
            frequency = n_pixels_per_class / n_image_pixels_with_class
            class_weighting = np.median(frequency) / frequency

        elif weight_mode == 'logarithmic':
            probabilities = n_pixels_per_class / np.sum(n_pixels_per_class)
            class_weighting = 1 / np.log(c + probabilities)

        if np.isnan(np.sum(class_weighting)):
            print(f"n_pixels_per_class: {n_pixels_per_class}")
            print(f"n_image_pixels_with_class: {n_image_pixels_with_class}")
            print(f"class_weighting: {class_weighting}")
            raise ValueError('class weighting contains NaNs')

        with open(class_weighting_filepath, 'wb') as f:
            pickle.dump(class_weighting, f)
        print(f'Saved class weights under {class_weighting_filepath}.')
        return class_weighting



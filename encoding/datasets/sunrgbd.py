# encoding:utf-8
import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
import pickle

image_w = 480
image_h = 480


__all__ = ['SUNRGBD', 'RandomHSV', 'scaleNorm', 'RandomScale', 'RandomCrop', 'RandomFlip', 'Normalize', 'ToTensor']


class SUNRGBD(Dataset):
    NUM_CLASS = 37
    CWD = os.getcwd()
    if CWD.endswith('Greene_Code'):
        BASE_DIR = '../dataset/sunrgbd/'
    else:
        BASE_DIR = '../../../dataset/sunrgbd/'
    #if os.path.dirname(CWD).endswith('experiments'):

    img_dir_train_file = os.path.join(BASE_DIR, 'data/img_dir_train.txt')
    depth_dir_train_file = os.path.join(BASE_DIR, 'data/depth_dir_train.txt')
    label_dir_train_file = os.path.join(BASE_DIR, 'data/label_train.txt')
    img_dir_test_file = os.path.join(BASE_DIR, 'data/img_dir_test.txt')
    depth_dir_test_file = os.path.join(BASE_DIR, 'data/depth_dir_test.txt')
    label_dir_test_file = os.path.join(BASE_DIR, 'data/label_test.txt')


    def __init__(self, transform=None, phase_train=True, data_dir=None):

        self.phase_train = phase_train
        self.transform = transform

        try:
            with open(self.img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            with open(self.depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(self.label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            with open(self.img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            with open(self.depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            with open(self.label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
            print('use old file ---')
        except:
            print('create new file')
            if data_dir is None:
                data_dir = self.BASE_DIR
            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data/', data_dir)

                depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)
                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path.replace(data_dir, ''))
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path.replace(data_dir, ''))
                    self.label_dir_train = np.append(self.label_dir_train, label_path.replace(data_dir, ''))
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path.replace(data_dir, ''))
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path.replace(data_dir, ''))
                    self.label_dir_test = np.append(self.label_dir_test, label_path.replace(data_dir, ''))

            local_file_dir = '/'.join(self.img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(self.img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(self.depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(self.label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(self.img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(self.depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(self.label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test

        label = np.load(os.path.join(self.BASE_DIR, label_dir[idx])).astype(np.int16) - 1     # -1 is void class
        depth = imageio.imread(os.path.join(self.BASE_DIR, depth_dir[idx]))
        image = imageio.imread(os.path.join(self.BASE_DIR, img_dir[idx]))

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def compute_class_weights(self, weight_mode='median_frequency', c=1.02):
        assert weight_mode in ['median_frequency', 'logarithmic', 'linear']

        # build filename
        class_weighting_filepath = os.path.join(self.BASE_DIR, 'weight', '{}.pickle'.format(weight_mode))

        if os.path.exists(class_weighting_filepath):
            class_weighting = pickle.load(open(class_weighting_filepath, 'rb'))
            print(f'Using {class_weighting_filepath} as class weighting')
            return class_weighting

        print('Compute class weights')

        n_pixels_per_class = np.zeros(self.NUM_CLASS+1)
        n_image_pixels_with_class = np.zeros(self.NUM_CLASS+1)
        label_dir = self.label_dir_train if self.phase_train else self.label_dir_test     # 只用训练数据计算weight

        for i in range(len(self)):
            label = np.load(os.path.join(self.BASE_DIR, label_dir[i])).astype(np.int16)
            h, w = label.shape
            current_dist = np.bincount(label.flatten(), minlength=self.NUM_CLASS+1)
            n_pixels_per_class += current_dist

            # For median frequency we need the pixel sum of the images where the specific class is present.
            # (It only matters if the class is present in the image and not how many pixels it occupies.)
            class_in_image = current_dist > 0
            n_image_pixels_with_class += class_in_image * h * w

            print(f'\r{i+1}/{len(self)}', end='')
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


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}




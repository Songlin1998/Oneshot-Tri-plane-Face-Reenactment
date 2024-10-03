import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from augmentations import AugmentationTransform
from PIL import ImageFile
import numpy as np
import json
import torch
import torchvision
import torchvision.transforms as transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

class Mydataset256(Dataset):
    def __init__(self, split, start, end, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = f'.../dataset/vox_crop_mini/train'
        elif split == 'test':
            self.ds_path = f'.../dataset/vox_crop_mini/test'
        else:
            raise NotImplementedError

        self.videos = sorted(glob.glob(self.ds_path+'/*'))[start:end]
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, vid_idx):

        video_path = self.videos[vid_idx]
        mask_path = os.path.join(video_path,'mask')
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)
        
        labels_file = os.path.join(video_path,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
         

        items = random.sample(list(range(nframes)), 2)
        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')
        camera_target = torch.from_numpy(labels[items[1]])
        img_source_256 = self.transform(img_source)
        img_target_256 = self.transform(img_target)
        img_target_128 = self.transform_128(img_target)

        img_name_target = frames_paths[items[1]].split('/')[-1].split('.')[0]
        mask_path_target = os.path.join(mask_path,img_name_target+'.png')
        mask_target = Image.open(mask_path_target).convert('RGB')
        mask_target = (self.transform(mask_target)+1)/2

        return img_source_256, img_target_256, camera_target, img_target_128, mask_target

    def __len__(self):
        return len(self.videos)

class Mydataset256_cross(Dataset):
    def __init__(self, split, start, end, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = f'.../dataset/vox_crop_mini/train'
        elif split == 'test':
            self.ds_path = f'.../dataset/vox_crop_mini/test'
        else:
            raise NotImplementedError

        self.videos = sorted(glob.glob(self.ds_path+'/*'))[start:end]
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, vid_idx):
        
        video_path_target = self.videos[vid_idx]
        frames_paths_target = sorted(glob.glob(video_path_target + '/*.png'))
        nframes_target = len(frames_paths_target)
        
        labels_file = os.path.join(video_path_target,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        
        items = random.sample(list(range(nframes_target)), 1)
        img_target = Image.open(frames_paths_target[items[0]]).convert('RGB')
        camera_target = torch.from_numpy(labels[items[0]])
        img_target_256 = self.transform(img_target)
        img_target_128 = self.transform_128(img_target)

        video_sample = random.sample(list(range(len(self.videos))), 1)
        video_path_source = self.videos[video_sample[0]]
        frames_paths_source = sorted(glob.glob(video_path_source + '/*.png'))
        nframes_source = len(frames_paths_source)
        
        items = random.sample(list(range(nframes_source)), 1)
        img_source = Image.open(frames_paths_source[items[0]]).convert('RGB')
        img_source_256 = self.transform(img_source)

        return img_source_256, img_target_256, camera_target, img_target_128

    def __len__(self):
        return len(self.videos)

class Mydataset256_cross_specific(Dataset):
    def __init__(self, split, start, end, source_id_path, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = f'.../dataset/vox_crop_mini/train'
        elif split == 'test':
            self.ds_path = f'.../dataset/vox_crop_mini/test'
        else:
            raise NotImplementedError

        self.videos = sorted(glob.glob(self.ds_path+'/*'))[start:end]
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        # source
        video_path_source = source_id_path
        self.frames_paths_source = sorted(glob.glob(video_path_source + '/*.png'))
        self.nframes_source = len(self.frames_paths_source)

    def __getitem__(self, vid_idx):
        
        # target
        items = random.sample(list(range(len(self.videos))), 1)
        video_path_target = self.videos[items[0]]
        frames_paths_target = sorted(glob.glob(video_path_target + '/*.png'))
        nframes_target = len(frames_paths_target)
        
        labels_file = os.path.join(video_path_target,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        
        items = random.sample(list(range(nframes_target)), 1)
        img_target = Image.open(frames_paths_target[items[0]]).convert('RGB')
        camera_target = torch.from_numpy(labels[items[0]])
        img_target_256 = self.transform(img_target)
        img_target_128 = self.transform_128(img_target)
        
        # source
        items = random.sample(list(range(self.nframes_source)), 1)
        img_source = Image.open(self.frames_paths_source[items[0]]).convert('RGB')
        img_source_256 = self.transform(img_source)

        return img_source_256, img_target_256, camera_target, img_target_128

    def __len__(self):
        return self.nframes_source

class Mydataset256_specific(Dataset):
    def __init__(self, split, start, end, source_id_path, transform=None, augmentation=False):

        
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        video_path = source_id_path
        self.frames_paths = sorted(glob.glob(video_path + '/*.png'))
        self.nframes = len(self.frames_paths)
        
        labels_file = os.path.join(video_path,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def __getitem__(self,idx):

        items = random.sample(list(range(self.nframes)), 2)
        img_source = Image.open(self.frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(self.frames_paths[items[1]]).convert('RGB')
        # camera_source = torch.from_numpy(labels[items[0]])
        camera_target = torch.from_numpy(self.labels[items[1]])
        img_source_256 = self.transform(img_source)
        img_target_256 = self.transform(img_target)
        # img_source_64 = self.transform_64(img_source)
        img_target_128 = self.transform_128(img_target)

        return img_source_256, img_target_256, camera_target, img_target_128

    def __len__(self):
        return self.nframes


class Vox256(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/vox/train'
        elif split == 'test':
            self.ds_path = './datasets/vox/test'
        else:
            raise NotImplementedError

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = './datasets/german/'
        self.driving_root = './datasets/vox/test/'

        self.anno = pd.read_csv('pairs_annotations/german_vox.csv')

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str('%03d' % self.anno['source'][idx])
        driving_name = self.anno['driving'][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)



class Vox256_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256_cross(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.anno = pd.read_csv('pairs_annotations/vox256.csv')
        self.transform = transform

    def __getitem__(self, idx):
        source_name = self.anno['source'][idx]
        driving_name = self.anno['driving'][idx]

        source_vid_path = os.path.join(self.ds_path, source_name)
        driving_vid_path = os.path.join(self.ds_path, driving_name)

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.videos)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/taichi/train/'
        else:
            self.ds_path = './datasets/taichi/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/taichi/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/'
        else:
            self.ds_path = './datasets/ted/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/ted/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)

class Mydataset256_ft(Dataset):
    def __init__(self, source_path, transform=None, augmentation=False):

        self.source_video_path = source_path
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        
        self.mask_path = os.path.join(self.source_video_path,'mask')
        self.frames_paths = sorted(glob.glob(self.source_video_path + '/*.png'))
        self.nframes = len(self.frames_paths)

        # 加载label文件
        labels_file = os.path.join(self.source_video_path,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __getitem__(self, vid_idx):

        items = random.sample(list(range(self.nframes)), 2)
        img_source = Image.open(self.frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(self.frames_paths[items[1]]).convert('RGB')
        # camera_source = torch.from_numpy(labels[items[0]])
        camera_target = torch.from_numpy(self.labels[items[1]])
        img_source_256 = self.transform(img_source)
        img_target_256 = self.transform(img_target)
        # img_source_64 = self.transform_64(img_source)
        img_target_128 = self.transform_128(img_target)

        img_name_target = self.frames_paths[items[1]].split('/')[-1].split('.')[0]
        mask_path_target = os.path.join(self.mask_path,img_name_target+'.png')
        mask_target = Image.open(mask_path_target).convert('RGB')
        mask_target = (self.transform(mask_target)+1)/2

        return img_source_256, img_target_256, camera_target, img_target_128, mask_target

    def __len__(self):
        return self.nframes

class Mydataset256_cross_ft_dual(Dataset):
    def __init__(self, source_video_path, target_video_path, transform=None, augmentation=False):

        self.videos = sorted(glob.glob('.../dataset/vox_crop_mini/train/*'))
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        # source
        self.frames_paths_source = sorted(glob.glob(source_video_path + '/*.png'))
        self.nframes_source = len(self.frames_paths_source)

        # target
        self.frames_paths_target = sorted(glob.glob(target_video_path + '/*.png'))
        self.nframes_target = len(self.frames_paths_target)

        # 加载label文件
        labels_file = os.path.join(target_video_path,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __getitem__(self, vid_idx):        
        
        # target
        items = random.sample(list(range(self.nframes_target)), 1)
        img_target = Image.open(self.frames_paths_target[items[0]]).convert('RGB')
        camera_target = torch.from_numpy(self.labels[items[0]])
        img_target_256 = self.transform(img_target)
        img_target_128 = self.transform_128(img_target)

        # source
        items = random.sample(list(range(self.nframes_source)), 1)
        img_source = Image.open(self.frames_paths_source[items[0]]).convert('RGB')
        img_source_256 = self.transform(img_source)

        return img_source_256, img_target_256, camera_target, img_target_128

    def __len__(self):
        return self.nframes_source

class Mydataset256_cross_ft(Dataset):
    def __init__(self, split, start, end, source_video_path, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = f'.../dataset/vox_crop_mini/train'
        elif split == 'test':
            self.ds_path = f'.../dataset/vox_crop_mini/test'
        else:
            raise NotImplementedError

        self.videos = sorted(glob.glob(self.ds_path+'/*'))[start:end]
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self._raw_labels = None
        self._use_labels = True
        self._label_shape = None

        self.transform_128 = torchvision.transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        # source
        self.frames_paths_source = sorted(glob.glob(source_video_path + '/*.png'))
        self.nframes_source = len(self.frames_paths_source)

    def __getitem__(self, vid_idx):
        
        # target
        video_path_target = self.videos[vid_idx]
        frames_paths_target = sorted(glob.glob(video_path_target + '/*.png'))
        nframes_target = len(frames_paths_target)
        
        # 加载label文件
        labels_file = os.path.join(video_path_target,'dataset.json')
        with open(labels_file, 'r') as f:
            file = json.load(f)
        labels = file['labels']
        labels = [labels[i][1] for i in range(len(labels))]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        
        items = random.sample(list(range(nframes_target)), 1)
        img_target = Image.open(frames_paths_target[items[0]]).convert('RGB')
        camera_target = torch.from_numpy(labels[items[0]])
        img_target_256 = self.transform(img_target)
        img_target_128 = self.transform_128(img_target)

        
        items = random.sample(list(range(self.nframes_source)), 1)
        img_source = Image.open(self.frames_paths_source[items[0]]).convert('RGB')
        img_source_256 = self.transform(img_source)

        return img_source_256, img_target_256, camera_target, img_target_128

    def __len__(self):
        return len(self.videos)
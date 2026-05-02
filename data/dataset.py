import os
import random
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image, ImageOps


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch


def augment(img_in, img_tar, img_en, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_en = ImageOps.flip(img_en)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_en = ImageOps.mirror(img_en)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_en = img_en.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_en, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, label_dir, data_dir, enhance_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        data_filenames = [join(data_dir, f) for f in listdir(data_dir) if is_image_file(f)]
        data_filenames.sort()
        label_filenames = [join(label_dir, f) for f in listdir(label_dir) if is_image_file(f)]
        label_filenames.sort()
        self.data_filenames = data_filenames
        self.label_filenames = label_filenames
        self.enhance_path = enhance_dir
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        label = load_img(self.label_filenames[index])
        data = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        k = random.randint(0,3)
        if k == 0:
            enhance_filenames = self.enhance_path + '/DDIM64_1/' + file
        if k == 1 :
            enhance_filenames = self.enhance_path + '/DDIM64_2/' + file
        if k == 2 :
            enhance_filenames = self.enhance_path + '/DDIM64_3/' + file
        if k == 3 :
            enhance_filenames = self.enhance_path + '/DDIM64_4/' + file

        enhance = load_img(enhance_filenames)

        label = label.resize((512, 512), Image.BICUBIC)
        data = data.resize((512, 512), Image.BICUBIC)
        enhance = enhance.resize((512, 512), Image.BICUBIC)
        if self.data_augmentation:
            data, label, enhance, _ = augment(data, label, enhance)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
            enhance = self.transform(enhance)

        return data, label, enhance, file

    def __len__(self):
        return len(self.data_filenames)


class DatasetFromFolderTest(data.Dataset):
    def __init__(self, data_dir, label_dir = None, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        input1 = input.resize((256,256), Image.BICUBIC)
        label = input.resize((64,64), Image.BICUBIC)

        if self.transform:
            input1 = self.transform(input1)
            label = self.transform(label)

        return input1, label, file

    def __len__(self):
        return len(self.data_filenames)


class Pr_DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super(Pr_DatasetFromFolder, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames

        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        label = load_img(self.label_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        input = input.resize((256,256), Image.BICUBIC)
        label = label.resize((256,256), Image.BICUBIC)

        if self.transform:
            input = self.transform(input)
            label = self.transform(label)

        return input, label, file

    def __len__(self):
        return len(self.data_filenames)
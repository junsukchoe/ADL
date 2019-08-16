import torch
import os
from PIL import Image
from torch.utils.data import Dataset

import random
from collections import Counter
import argparse


class CUBClsDataset(Dataset):
    def __init__(self, root=None, datalist=None, transform=None):

        self.root = root
        self.datalist = datalist
        self.transform = transform
        image_ids = []
        image_names= []
        image_labels = []
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_ids.append(int(info[0]))
                image_names.append(info[1])
                image_labels.append(int(info[2]))
        self.image_ids = image_ids
        self.image_names = image_names
        self.image_labels = image_labels

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_label = self.image_labels[idx]
        image = Image.open(os.path.join(self.root, 'images/', image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label

    def __len__(self):
        return len(self.image_ids)


class CUBCamDataset(Dataset):
    def __init__(self, root=None, datalist=None, transform=None):

        self.root = root
        self.datalist = datalist
        self.transform = transform
        image_ids = []
        image_names= []
        image_labels = []
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_ids.append(int(info[0]))
                image_names.append(info[1])
                image_labels.append(int(info[2]))
        self.image_ids = image_ids
        self.image_names = image_names
        self.image_labels = image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_name = self.image_names[idx]
        image_label = self.image_labels[idx]
        image = Image.open(os.path.join(self.root, 'images/', image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


def load_image_path(dataset_path):
    '''
    return dict{image_id : image_path}
    '''
    image_paths = {}
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            image_id, image_path = line.strip().split()
            image_paths[image_id] = image_path
    return image_paths


def load_image_label(dataset_path):
    """
    return dict{image_id : image_label}
    """
    image_labels = {}
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            image_id, label_id = line.strip().split()
            image_labels[image_id] = int(label_id) - 1
    return image_labels


def create_image_sizes_file(dataset_path):
    '''
    save 'sizes.txt' in forms of
    [image_id] [width] [height]
    '''
    import cv2

    image_paths = load_image_path(dataset_path)
    image_sizes = []
    for image_id, image_path in image_paths.items():
        image = cv2.imread(os.path.join(dataset_path, 'images', image_path))
        image_sizes.append([image_id, image.shape[1], image.shape[0]])
    with open(os.path.join(dataset_path, 'sizes.txt'), 'w') as f:
        for image_id, w, h in image_sizes:
            f.write("%s %d %d\n" % (str(image_id), w, h))


def load_train_test_split(dataset_path):
    '''
    return dict{image_id: 1 for train
                          0 for test }
    '''
    image_train_test = {}
    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            image_id, is_train = line.strip().split()
            image_train_test[image_id] = is_train
    return image_train_test


def split_dataset(dataset_path, fraction_per_class=0.1, shuffle=False):
    '''
    save 'train.txt', 'val.txt', 'test.txt'
    in fomrs of
    [image_id] [image_name(path)] [image_label]
    '''
    image_names = load_image_path(dataset_path)
    image_labels = load_image_label(dataset_path)
    image_train_test = load_train_test_split(dataset_path)
    num_train, num_val, num_test = 0, 0, 0
    # train_image_ids = []
    # for image_id in image_names.keys():
    #     if image_train_test[image_id] == '1':
    #         train_image_ids.append(image_id)
    #
    # subset_train_image_ids = []
    # val_image_ids = []
    #
    # class_labels = [image_labels[image_id] for image_id in train_image_ids]
    # num_image_per_label = Counter(class_labels)
    # num_val_image_per_label = {label: 0 for label in num_image_per_label.keys()}
    #
    # for label, num_image in num_image_per_label.items():
    #     if num_image <= 1:
    #         print("Warning: label %d has only %d images" %(label, image_count))
    #
    # if shuffle:
    #     random.shuffle(train_image_ids)
    #
    # for image_id in train_image_ids:
    #     image_label = image_labels[image_id]
    #
    #     if num_val_image_per_label[image_label] < num_image_per_label[image_label] * fraction_per_class:
    #         val_image_ids.append(image_id)
    #         num_val_image_per_label[image_label] += 1
    #     else:
    #         subset_train_image_ids.append(image_id)
    #
    # with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
    #     for image_id in subset_train_image_ids:
    #         f.write("%s %s %s\n" % (image_id, image_names[image_id], image_labels[image_id]))
    #         num_train += 1
    # with open(os.path.join(dataset_path, 'val.txt'), 'w') as f:
    #     for image_id in val_image_ids:
    #         f.write("%s %s %s\n" % (image_id, image_names[image_id], image_labels[image_id]))
    #         num_val += 1
    with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
        for image_id in image_names.keys():
            if image_train_test[image_id] == '1':
                f.write("%s %s %s\n" % (image_id, image_names[image_id], image_labels[image_id]))
                num_train += 1
    with open(os.path.join(dataset_path, 'test.txt'), 'w') as f:
        for image_id in image_names.keys():
            if image_train_test[image_id] == '0':
                f.write("%s %s %s\n" % (image_id, image_names[image_id], image_labels[image_id]))
                num_test += 1

    print('%d train data is saved in \'%s\\train.txt\'.\n' % (num_train, dataset_path))
    # print('%d val data is saved in \'%s\\val.txt\'.\n' % (num_val, dataset_path))
    print('%d test data is saved in \'%s\\test.txt\'.\n' % (num_test, dataset_path))


def get_image_name(file_path):
    '''
    return dict{image_id: image name}
    '''
    image_names = {}
    with open(file_path) as f:
        for line in f:
            image_id, image_name, image_label = line.strip().split()
            image_names[int(image_id)] = image_name
    return image_names


parser = argparse.ArgumentParser(description='CUB dataset list generation.')
parser.add_argument('data', metavar='DIR', help='path to dataset')

def main():
    # dataset_path = 'datasets/CUB_200_2011/CUB_200_2011'
    # split_dataset(dataset_path)
    args = parser.parse_args()
    split_dataset(args.data)
    # create_image_sizes_file(dataset_path)

if __name__ == "__main__":

    main()

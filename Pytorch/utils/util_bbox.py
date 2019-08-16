import os
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.dataset_cub import CUBCamDataset, get_image_name
import cv2

def load_image_size(dataset_path='datalist'):

    image_sizes = {}
    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])
            image_width, image_height = map(float, file_info[1:])

            image_sizes[image_id] = [image_width, image_height]
    return image_sizes
def load_bbox_size(dataset_path='datalist',
                   resize_size=256, crop_size=224):
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}
    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])

            x, y, bbox_width, bbox_height =  map(float, file_info[1:])

            origin_bbox[image_id] = [x, y, bbox_width, bbox_height]

    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])
            image_width, image_height = map(float, file_info[1:])

            image_sizes[image_id] = [image_width, image_height]

    resize_size = float(resize_size-1)
    shift_size = (resize_size - crop_size) // 2
    for i in origin_bbox.keys():
        x, y, bbox_width, bbox_height = origin_bbox[i]
        image_width, image_height = image_sizes[i]
        left_bottom_x = x / image_width * resize_size - shift_size
        left_bottom_y = y / image_height * resize_size - shift_size

        right_top_x = (x+bbox_width) / image_width * resize_size - shift_size
        right_top_y = (y+bbox_height) / image_height * resize_size - shift_size
        resized_bbox[i] = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]


    return resized_bbox

def see_originial_bouding_box(dataset_path='datalist'):
    cls_img_path  = {}
    image_sizes = {}
    origin_bbox = {}
    with open(os.path.join(dataset_path, 'val.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            file_name = file_info[1]
            file_id = int(file_info[0])
            cls_img_path[file_id] = file_name
    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])
            image_width, image_height = map(float, file_info[1:])

            image_sizes[image_id] = [image_width, image_height]
    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])

            x, y, bbox_width, bbox_height =  map(float, file_info[1:])

            origin_bbox[image_id] = [int(x), int(y), int(bbox_width), int(bbox_height)]

    for i in sorted(cls_img_path.keys()):
        gxa = origin_bbox[i][0]
        gya = origin_bbox[i][1]
        gxb = origin_bbox[i][0] + origin_bbox[i][2]
        gyb = origin_bbox[i][1] + origin_bbox[i][3]
        image = cv2.imread(os.path.join(dataset_path,'images/',cls_img_path[i]), cv2.IMREAD_COLOR)
        cv2.rectangle(image, (gxa,gya), (gxb, gyb), (0, 0, 255), 2)
        file_name = cls_img_path[i].strip().split('/')[1]

        height, width, channel = image.shape
        # print(width == image_sizes[i][0], height == image_sizes[i][1])
        # print(i, height, image_sizes[i][0], width, image_sizes[i][1])

        # cv2.imwrite('./result_BBOX/{}'.format(file_name), image)
def see_transformed_bounding_box(dataset_path='/workspace/TPAMI2019/CUB_200_2011/CUB_200_2011'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    testset = CUBCamDataset(dataset_path, 'val.txt', transforms=transforms_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
    name= 0
    bbox = load_bbox_size()
    image_names = get_image_name(dataset_path, 'val.txt')

    for i, (images, target, images_id) in enumerate(val_loader):
        images = ((images * 0.22 + 0.45) * 255.0).cpu().detach().numpy().transpose([0, 2, 3, 1])[..., ::-1]
        images = images - np.min(images)
        images = images / np.max(images) * 255.0

        for j in range(target.size(0)):

            image = images[j]
            image_id = images_id[j].item()
            gxa = int(bbox[image_id][0])
            gya = int(bbox[image_id][1])
            gxb = int(bbox[image_id][2])
            gyb = int(bbox[image_id][3])
            # print(gxa, gya, gxb, gyb)
            # print('hello')
            image = cv2.rectangle(image, (gxa,gya), (gxb, gyb), (0, 0, 255), 2)
            # cv2.imwrite('./result_resized/{}'.format(str(name)+'.jpg'), image)
            cv2.imwrite('./result_resized/'+image_names[image_id].split('/')[1], image)
            name += 1

def main():

    see_transformed_bounding_box()
    # see_originial_bouding_box()
    #
    # for i in bbox.keys():
    #     print(bbox[i])

if __name__ == '__main__':
    main()

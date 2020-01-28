import cv2
import numpy as np
import os

from tensorpack import PredictConfig
from tensorpack import get_model_loader
from tensorpack import SimpleDatasetPredictor
from tensorpack.utils import viz
from tensorpack.utils.fs import mkdir_p

from data_loader import get_data
from os.path import join as ospj
import dataflow


def get_model(model, ckpt_name, option):
    model_path = ospj('train_log', option.log_dir, ckpt_name)
    ds = get_data('val', option)
    pred_config = PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['input', 'label', 'bbox'],
        output_names=['wrong-top1', 'top5', 'actmap', 'grad'],
        return_input=True
    )

    return SimpleDatasetPredictor(pred_config, ds)


def get_meta(option):
    if option.dataset_name == 'ILSVRC':
        meta = dataflow.ImagenetMeta().get_synset_words_1000()
        meta_labels = dataflow.ImagenetMeta().get_synset_1000()
    elif option.dataset_name == 'CUB':
        meta = dataflow.CUBMeta().get_synset_words_1000()
        meta_labels = dataflow.CUBMeta().get_synset_1000()
    else:
        raise KeyError("Unavailable dataset: {}".format(option.dataset_name))

    return meta, meta_labels


def get_log_dir(option):
    threshold_idx = int(option.cam_threshold * 100)
    dirname = ospj('train_log', option.log_dir, 'result', str(threshold_idx))
    if not os.path.isdir(dirname):
        mkdir_p(dirname)
    return dirname, threshold_idx


def get_cam(index, averaged_gradients, convmaps, option):
    batch_size, channel_size, height, width = np.shape(convmaps)

    averaged_gradient = averaged_gradients[index]
    convmap = convmaps[index, :, :, :]
    mergedmap = np.matmul(averaged_gradient,
                          convmap.reshape((channel_size, -1))). \
        reshape(height, width)
    mergedmap = cv2.resize(mergedmap,
                           (option.final_size, option.final_size))
    heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
    return heatmap


def get_estimated_box(heatmap, option):
    gray_heatmap = cv2.cvtColor(heatmap.astype('uint8'), cv2.COLOR_RGB2GRAY)
    threshold_value = int(np.max(gray_heatmap) * option.cam_threshold)

    _, thresholded_gray_heatmap = cv2.threshold(gray_heatmap, threshold_value,
                                                255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(thresholded_gray_heatmap,
                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0, 0, 1, 1]

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    estimated_box = [x, y, x + w, y + h]

    return estimated_box


def get_gt_box(index, bbox):
    gt_x_a = int(bbox[index][0][0])
    gt_y_a = int(bbox[index][0][1])
    gt_x_b = int(bbox[index][1][0])
    gt_y_b = int(bbox[index][1][1])

    gt_box = [gt_x_a, gt_y_a, gt_x_b, gt_y_b]
    return gt_box


def draw_images_with_boxes(index, images, heatmap, estimated_box, gt_box):
    image_with_bbox = images[index].astype('uint8')
    cv2.rectangle(image_with_bbox, (estimated_box[0], estimated_box[1]),
                  (estimated_box[2], estimated_box[3]), (0, 255, 0), 2)
    cv2.rectangle(image_with_bbox, (gt_box[0], gt_box[1]),
                  (gt_box[2], gt_box[3]), (0, 0, 255), 2)
    blend = images[index] * 0.5 + heatmap * 0.5
    blend = blend.astype('uint8')
    heatmap = heatmap.astype('uint8')
    concat = np.concatenate((image_with_bbox, heatmap, blend), axis=1)
    return image_with_bbox, concat


def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = np.maximum(0, (x_b - x_a + 1)) * np.maximum(0, (
            y_b - y_a + 1))
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


class LocEvaluator(object):
    def __init__(self):
        self.cnt = 0
        self.cnt_false = 0
        self.hit_known = 0
        self.hit_top1 = 0

        self.top1_cls = None
        self.gt_known_loc = None
        self.top1_loc = None

    def accumulate_acc(self, estimated_box, gt_box, wrongs, index):
        iou = compute_iou(estimated_box, gt_box)
        if wrongs[index]:
            self.cnt_false += 1
        if iou > 0.5 or iou == 0.5:
            self.hit_known += 1
        if (iou > 0.5 or iou == 0.5) and not wrongs[index]:
            self.hit_top1 += 1
        self.cnt += 1

    def compute_acc(self):
        self.top1_cls = 1 - self.cnt_false / self.cnt
        self.gt_known_loc = self.hit_known / self.cnt
        self.top1_loc = self.hit_top1 / self.cnt

    def save_img(self, threshold_idx, classname, concat, option):
        fname = 'train_log/{}/result/{}/cam{}-{}.jpg'.format(
            option.log_dir, threshold_idx, self.cnt, classname)
        cv2.imwrite(fname, concat)

    def print_acc(self, threshold_idx, option):
        fname = 'train_log/{}/result/{}/Loc.txt'. \
            format(option.log_dir, threshold_idx)
        with open(fname, 'w') as f:
            line = 'cls: {}\ngt_loc: {}\ntop1_loc: {}'. \
                format(self.top1_cls, self.gt_known_loc, self.top1_loc)
            f.write(line)
        print('thr: {}\n'.format(float(threshold_idx) / 100) + line)


def evaluate(model, ckpt_name, option):
    pred = get_model(model, ckpt_name, option)
    meta, meta_labels = get_meta(option)
    dirname, threshold_idx = get_log_dir(option)

    evaluator = LocEvaluator()

    for inputs, outputs in pred.get_result():
        images, labels, bbox = inputs
        wrongs, top5, convmaps, gradients = outputs

        if option.is_data_format_nhwc:
            convmaps = np.transpose(convmaps, [0, 3, 1, 2])
            gradients = np.transpose(gradients, [0, 3, 1, 2])

        averaged_gradients = np.mean(gradients, axis=(2, 3))

        for i in range(np.shape(convmaps)[0]):
            heatmap = get_cam(i, averaged_gradients, convmaps, option)
            estimated_box = get_estimated_box(heatmap, option)
            gt_box = get_gt_box(i, bbox)
            evaluator.accumulate_acc(estimated_box, gt_box, wrongs, i)
            bbox_img, concat = draw_images_with_boxes(i, images, heatmap,
                                                      estimated_box, gt_box)
            cls_name = meta[meta_labels[labels[i]]].split(',')[0]

            if evaluator.cnt < 500:
                evaluator.save_img(threshold_idx, cls_name, concat, option)

            if evaluator.cnt == option.number_of_val:
                evaluator.compute_acc()
                evaluator.print_acc(threshold_idx, option)
                return


def evaluate_wsol(option, model, interval=False):
    option.batch_size = 100
    if interval:
        for i in range(option.number_of_cam_curve_interval):
            option.cam_threshold = 0.05 + i * 0.05
            evaluate(model, 'min-val-error-top1.index', option)
    else:
        evaluate(model, 'min-val-error-top1.index', option)

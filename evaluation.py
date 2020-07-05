# -*- coding:utf-8 -*-
import os
import glob
import re
import argparse
from tqdm import tqdm 

import torch
from torchvision import transforms
from dataset import ICDARDataSet
from torch.utils import data

import numpy as np
from PIL import Image, ImageDraw
import shapely.geometry as plg

from model import EAST
from dataset import get_rotate_mat
import lanms


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(points[0])
    res_boxes[0, 4] = int(points[1])
    res_boxes[0, 1] = int(points[2])
    res_boxes[0, 5] = int(points[3])
    res_boxes[0, 2] = int(points[4])
    res_boxes[0, 6] = int(points[5])
    res_boxes[0, 3] = int(points[6])
    res_boxes[0, 7] = int(points[7])
    point_mat = res_boxes[0].reshape([2, 4]).T
    return plg.Polygon(point_mat)


def rectangle_to_polygon(rect):
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(rect.xmin)
    res_boxes[0, 4] = int(rect.ymax)
    res_boxes[0, 1] = int(rect.xmin)
    res_boxes[0, 5] = int(rect.ymin)
    res_boxes[0, 2] = int(rect.xmax)
    res_boxes[0, 6] = int(rect.ymin)
    res_boxes[0, 3] = int(rect.xmax)
    res_boxes[0, 7] = int(rect.ymax)

    point_mat = res_boxes[0].reshape([2, 4]).T

    return plg.Polygon(point_mat)


def rectangle_to_points(rect):
    points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin),
              int(rect.xmin), int(rect.ymin)]
    return points


def get_union(p_d, p_g):
    area_a = p_d.area
    area_b = p_g.area
    return area_a + area_b - get_intersection(p_d, p_g)


def get_intersection_over_union(p_d, p_g):
    try:
        return get_intersection(p_d, p_g) / get_union(p_d, p_g)
    except:
        return 0


def get_intersection(p_d, p_g):
    if not p_d.intersects(p_g):
        return 0
    p_int = p_d & p_g
    return p_int.area


def compute_ap(conf_list, match_list, num_gt_care):
    correct = 0
    ap = 0
    if len(conf_list) > 0:
        conf_list = np.array(conf_list)
        match_list = np.array(match_list)
        sorted_ind = np.argsort(-conf_list)
        conf_list = conf_list[sorted_ind]
        match_list = match_list[sorted_ind]
        for n in range(len(conf_list)):
            match = match_list[n]
            if match:
                correct += 1
                ap += float(correct) / (n + 1)

        if num_gt_care > 0:
            ap /= num_gt_care

    return ap


def compute_metric(gt_boxes, gt_transcriptions, predict_boxes, evaluation_params):
    recall, precision, hmean = (0, 0, 0)
    det_matched, num_gt_care, num_det_care = (0, 0, 0)

    gt_pols = []
    det_pols = []

    # Array of Ground Truth Polygons' keys marked as don't Care
    gt_dont_care_pols_num = []
    # Array of Detected Polygons' matched with a don't Care GT
    det_dont_care_pols_num = []

    pairs = []
    det_matched_nums = []

    for n in range(len(gt_boxes)):
        points = gt_boxes[n]
        transcription = gt_transcriptions[n]
        dont_care = transcription == "###"
        gt_pol = polygon_from_points(points)
        gt_pols.append(gt_pol)
        if dont_care:
            gt_dont_care_pols_num.append(len(gt_pols) - 1)

    num_gt_care = (len(gt_pols) - len(gt_dont_care_pols_num))

    if predict_boxes is not None:
        for n in range(len(predict_boxes)):
            points = predict_boxes[n]
            det_pol = polygon_from_points(points)
            det_pols.append(det_pol)
            if len(gt_dont_care_pols_num) > 0:
                for dont_care_pol in gt_dont_care_pols_num:
                    dont_care_pol = gt_pols[dont_care_pol]
                    intersected_area = get_intersection(dont_care_pol, det_pol)
                    pd_dimensions = det_pol.area
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                    if precision > evaluation_params['AREA_PRECISION_CONSTRAINT']:
                        det_dont_care_pols_num.append(len(det_pols) - 1)
                        break

    if len(gt_pols) > 0 and len(det_pols) > 0:
        # Calculate IoU and precision matrixs
        output_shape = [len(gt_pols), len(det_pols)]
        iou_mat = np.empty(output_shape)
        gt_rect_mat = np.zeros(len(gt_pols), np.int8)
        det_rect_mat = np.zeros(len(det_pols), np.int8)
        for gt_num in range(len(gt_pols)):
            for det_num in range(len(det_pols)):
                p_g = gt_pols[gt_num]
                p_d = det_pols[det_num]
                iou_mat[gt_num, det_num] = get_intersection_over_union(p_d, p_g)

        for gt_num in range(len(gt_pols)):
            for det_num in range(len(det_pols)):
                if gt_rect_mat[gt_num] == 0 and det_rect_mat[det_num] == 0 and gt_num not in gt_dont_care_pols_num \
                        and det_num not in det_dont_care_pols_num:
                    if iou_mat[gt_num, det_num] > evaluation_params['IOU_CONSTRAINT']:
                        gt_rect_mat[gt_num] = 1
                        det_rect_mat[det_num] = 1
                        det_matched += 1
                        pairs.append({'gt': gt_num, 'det': det_num})
                        det_matched_nums.append(det_num)

        num_det_care = (len(det_pols) - len(det_dont_care_pols_num))
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
        else:
            recall = float(det_matched) / num_gt_care
            precision = 0 if num_det_care == 0 else float(det_matched) / num_det_care

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    metrics = {'precision': precision, 'recall': recall, 'hmean': hmean,
               'num_care_gt': num_gt_care, 'num_care_predict': num_det_care, 'matched': det_matched}
    return metrics


def compute_total_metric(metrics):
    num_of_gt, num_of_detection, num_of_matched = (0, 0, 0)
    for metric in metrics.values():
        num_of_gt += metric['num_care_gt']
        num_of_detection += metric['num_care_predict']
        num_of_matched += metric['matched']
    recall = 0 if num_of_gt == 0 else float(num_of_matched) / num_of_gt
    precision = 0 if num_of_detection == 0 else float(num_of_matched) / num_of_detection
    hmean = 0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)

    return {'precision': precision, 'recall': recall, 'hmean': hmean}


def resize_img(img):
    """
    resize image to be divisible by 32
    """
    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w
    return img, ratio_h, ratio_w


def load_pil(img):
    """convert PIL Image to torch.Tensor
    """
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
    """check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    """
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    """restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """
    get boxes from feature map
    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polys <numpy.ndarray, (n,9)>
    """
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    ret_boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    if ret_boxes is None:
        print('it')
    return ret_boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    """refine boxes
    Input:
        boxes  : detected polys <numpy.ndarray, (n,9)>
        ratio_w: ratio of width
        ratio_h: ratio of height
    Output:
        refined boxes
    """
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def plot_boxes(img, boxes):
    """plot boxes on image
    """
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 255, 0))
    return img


def read_boxes_from_file(file_path):
    points_array = []
    transcription_array = []
    with open(file_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        m = re.match(
            r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',
            line)
        if m is None:
            raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription")
        points = [float(m.group(i)) for i in range(1, 9)]
        transcription = m.group(9)
        m2 = re.match(r'^\s*\"(.*)\"\s*$', transcription)
        if m2 is not None:  # Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")
        points_array.append(points)
        transcription_array.append(transcription)
    return points_array, transcription_array


# def evaluate(model, data_path):
def evaluate(model, image_path, gt_path):
    """
    image path를 하나씩 읽어서 하나씩 evaluation
    정확하게 동작하는 지 확인 후 batch evaluation 구현
    :param model: EAST model object
    :param image_path: evaluation target image path
    :param gt_path: evaluation ground truth text file path
    :return: None
    """
    # image_path = os.path.join(data_path, 'img')
    # gt_path = os.path.join(data_path, 'gt')
    model.eval()

    eval_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    evaluation_config = {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'CONFIDENCES': False,  # Detections must include confidence value. AP will be calculated
    }

    sample_metrics = {}
    for sample_path in glob.glob(image_path + '/*.jpg'):
        file_name = os.path.basename(sample_path)
        gt_file_path = os.path.join(gt_path, 'gt_' + file_name.replace('.jpg', '.txt'))
        gt_boxes, gt_transcriptions = read_boxes_from_file(gt_file_path)
        sample_img = Image.open(sample_path)
        sample_img, ratio_h, ratio_w = resize_img(sample_img)
        sample_img = eval_transform(sample_img).unsqueeze(0).cuda()

        with torch.no_grad():
            score, geo = model(sample_img)
        # box NMS 중에 box index 정보가 없어지므로 confidence score 구할 수 없다.
        # box NMS (local NMS) code 분석 후 이 정보도 함께 넘겨 받기 전에는 confidence score 없이 간다.
        # 즉 AP 값을 구할 수 없다.
        predict_boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
        predict_boxes = adjust_ratio(predict_boxes, ratio_w, ratio_h)
        per_sample_metric = compute_metric(gt_boxes, gt_transcriptions, predict_boxes, evaluation_config)
        sample_metrics[gt_file_path] = per_sample_metric        
    total_metric = compute_total_metric(sample_metrics)
    return sample_metrics, total_metric

def print_metrics(metrics):
    for metric in metrics:
        print (str(metric))

def evaluate_batch(model, config):
    """
    image path를 하나씩 읽어서 하나씩 evaluation
    정확하게 동작하는 지 확인 후 batch evaluation 구현
    :param model: EAST model object
    :param config: evaluation ground truth text file path
    :return: None
    """
    model.eval()
    eval_dataset = ICDARDataSet(config.eval_data_path, mode='eval', heigth=704)
    evaluation_config = {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'CONFIDENCES': False,  # Detections must include confidence value. AP will be calculated
    }

    data_loader = data.DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False,
                                  num_workers=config.num_workers, drop_last=False)
    sample_metrics = {}
    for img, indices, ratios_w, ratios_h in tqdm(data_loader, desc='Evaluation...'):
        img = img.cuda()
        with torch.no_grad():
            scores, geos = model(img)

        for i, index in enumerate(indices):
            ratio_w = ratios_w[i]
            ratio_h = ratios_h[i]
            score = scores[i]
            geo = geos[i]
            
            predict_boxes = get_boxes(score.cpu().numpy(), geo.cpu().numpy())
            predict_boxes = adjust_ratio(predict_boxes, ratio_w, ratio_h)
            gt_boxes, gt_transcriptions = eval_dataset.get_gt_for_eval(index)
            per_sample_metric = compute_metric(gt_boxes, gt_transcriptions, predict_boxes, evaluation_config)
            sample_metrics[eval_dataset.get_gt_file_name(index)] = per_sample_metric            
   
    total_metric = compute_total_metric(sample_metrics)
    return sample_metrics, total_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_path', type=str, default='../ICDAR_2015/test')
    parser.add_argument('--out', type=str, default='pths')
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='pths/east_vgg16.pth')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    east = EAST().to(device)
    east.load_state_dict(torch.load(args.model_path))

    data_parallel = False
    if torch.cuda.device_count() > 1:
        east = torch.nn.DataParallel(east)
        data_parallel = True

    single_metrics, total_metric = evaluate_batch(east, args)
    print(single_metrics)
    print(total_metric)
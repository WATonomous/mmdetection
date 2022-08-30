"""
The purpose of this script is to generate annotations for inactive agents of ROAD dataset.

"""

import pickle
import json
import cv2

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# Detection results on ROAD dataset by the COCO pretrained model
with open('./road_annotations/coco_detection_on_train2_quarter.pkl', 'rb') as f:
    coco_data = pickle.load(f)

# Ground-truth of ROAD dataset with COCO format annotation
with open('./road_annotations/coco_annotation_train2_quarter.json', 'rb') as f:
    road_data = json.load(f)

image_num = len(road_data['images'])

# ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL']
cats = [1,4,5,6,7]
tmp = list()
for i in range(image_num):
    tmp.append(list())

road_data_anno = road_data['annotations']
anno_num = len(road_data_anno)
ann_id = anno_num

for idx, anno in enumerate(road_data_anno):
    image_id = anno['image_id']
    anno_id = anno['id']
    assert anno_id == idx
    tmp[image_id].append(anno_id)

for image_id, xx in enumerate(coco_data):
    img_path = road_data['images'][image_id]['file_name']
    if image_id % 1000 == 0:
        print(image_id)
    # print(img_path)
    image = cv2.imread(img_path)
    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    color3 = (0, 0, 255)
    thickness = 2
    
    save_img = 'vis_{0}.jpg'.format(image_id)

    det_box = []
    gt_box = []

    for class_id, dets in enumerate(xx):
        # class car         
        if dets.shape[0] != 0 and class_id == 2:
            for det in dets:
                score = det[4]
                start_point = (int(det[0]), int(det[1]))
                end_point = (int(det[2]), int(det[3]))
                if score > 0.7 and (det[2]-det[0]) < 640 and (end_point[1] < 840):
                    # COCO detections
                    image = cv2.rectangle(image, start_point, end_point, color1, thickness)
                    det_box.append(det)
            anno_ids = tmp[image_id]

            for anno_id in anno_ids:
                category = road_data_anno[anno_id]['category_id']
                bbox = road_data_anno[anno_id]['bbox']
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
                if (category in cats):
                    image = cv2.rectangle(image, start_point, end_point, color2, thickness) 
                    # GT Annotations
                    gt_box.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

    new_det_box = []
    for dbox in det_box:
        max_overlap = 0
        for gbox in gt_box:
            overlap = bb_intersection_over_union(dbox, gbox)
            if ( overlap > max_overlap):
                max_overlap = overlap
        if max_overlap < 0.2:
            new_det_box.append(dbox)

    for det in new_det_box:
        start_point = (int(det[0]), int(det[1]))
        end_point = (int(det[2]), int(det[3]))
        image = cv2.rectangle(image, start_point, end_point, color3, thickness)

    # Visualization of the new annotations
    # if image_id % 100 == 0:
    # if True:
        # cv2.imwrite(save_img,image)

    for box in new_det_box:
        box = box[0:4].tolist()
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        area = box[2] * box[3]
        anno_info = dict(
            iscrowd = 0,
            category_id = 10,                        
            bbox = box,
            area = area,
            image_id = image_id,
            id = ann_id
        )
        assert len(box) == 4
        road_data_anno.append(anno_info)
        ann_id += 1


print(ann_id - 1)
road_data['annotations'] = road_data_anno
categories = road_data['categories']
categories.append({'id':10, 'name': 'Inactive'})
road_data['categories'] = categories
print(len(road_data_anno))

# save path for the new annotations
out_json = open('./road_annotations/coco_annotation_train2_quarter_inactive.json', 'w')
json.dump(road_data, out_json)


import pickle
import json

#f = open('./frcnn_x101_val1.pkl','rb')
#f = open('./frcnn_x101_val1_inactive.pkl','rb')
f = open('./work_dirs/fpn_x101_64x4d_finetune_inactive_merged_config_old/frcnn_x101_val1_inactive_merged_new.pkl','rb')
#f = open('./flow_color_model_fpn_x101_64x4d_finetune_inactive_merged_results.pkl','rb')
data = pickle.load(f)

#f2 = open('/home/liqq/project/699/road-dataset/coco_annotation_val1.json', 'r')
#f2 = open('/home/liqq/project/699/road-dataset/coco_annotation_val1_inactive.json', 'r')
f2 = open('./coco_annotation_val1_new.json', 'r')
gt = json.load(f2)

#output_path = "val1_detections.jsonl"
#output_path = "val1_detections_inactive.jsonl"
#output_path = "val1_detections_inactive_merged.jsonl"
output_path = "./work_dirs/fpn_x101_64x4d_finetune_inactive_merged_config_old/val1_detections_inactive_merged_new.jsonl"

num = len(gt['images'])
print(f'num = {num}')
print(f'data: {len(data)}')
for i in range(num):
    img_path = gt['images'][i]['file_name']
    frame_id = int(img_path.split('/')[-1].split('.')[0])
    video_name = img_path.split('/')[-2]
    frame_name = '{0}.{1}'.format(video_name, frame_id)
    dets = data[i]
    detections = []
    for class_id, det in enumerate(dets):
        # if len(det) != 0:
        if len(det) != 0 and class_id != 5:
            for box in det:
                bbox = box[0:4].tolist()
                score = box[4].tolist()
                detections.append({'bbox': bbox, 'score': score, 'class': class_id})
    output_obj = {
        'frameName': frame_name,
        'detections': detections
    }
    # if len(detections) > 0:
    with open(output_path, 'a') as appender:
        appender.write(f"{json.dumps(output_obj)}\n")

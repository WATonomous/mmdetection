import json
import numpy as np

def detection_bbox_to_ava(bbox):
    x1, y1, x2, y2 = bbox
    convbb = [x1/1280, y1/960, x2/1280, y2/960]
    return convbb

annotation_path = './flow_color_model_fpn_x101_64x4d_finetune_val1_detections_inactive_merged.jsonl'
output_path = './flow_color_model_fpn_x101_64x4d_finetune_val1_detections_inactive_merged_gt_format.json'
gt_path = '/road/road_trainval_v1.0.json'

with open(gt_path, 'r') as f:
    gt = json.load(f)

box_count = 0
with open(annotation_path, "r") as f:
    json_list = list(f)
    total_num = len(json_list)
    db = {}
    for idx, keyframe_str in enumerate(json_list):
        keyframe_dict = json.loads(keyframe_str)
        video = keyframe_dict['frameName'].split(".")[0]
        frame_id = int(keyframe_dict['frameName'].split(".")[1])
        labels = keyframe_dict["detections"]
        print(f'{idx}/{total_num}')
        # print(video)
        # print(frame_id)
        # print(labels)
        frame = {}
        frame['annotated'] = 1
        frame['input_image_id'] = frame_id
        frame['width'] = 1280
        frame['height'] = 960
        annos = {}
        for label in labels:
            det = {
                'box': detection_bbox_to_ava(label['bbox']),
                'agent_ids': [label['class']],
                'action_ids': [1], # fake action ids
                'score': label['score']
            }
            # Set threshold
            if det['score'] > 0.3:
                annos[str(box_count)] = det
                box_count += 1
        frame['annos'] = annos
        if video not in db.keys():
            db[video] = {}
            db[video]['frames'] = {}
            db[video]['numf'] = gt['db'][video]['numf']
            db[video]['split_ids'] = gt['db'][video]['split_ids']
        db[video]['frames'][str(frame_id)] = frame
    output = {}
    output['db'] = db
    # json_string = json.dumps(output)
    with open(output_path, 'w') as outfile:
        json.dump(output, outfile)






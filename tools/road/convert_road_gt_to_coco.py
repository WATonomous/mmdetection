"""
The purpose of this script is to convert annotation of ROAD dataset to COCO dataset format.

"""

import json, pdb, argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cmx
import matplotlib.colors as colors
from PIL import Image
import numpy as np
import os


def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids                

if __name__ == '__main__':

    # save path for the coco format annotation
    is_train_set = True
    if is_train_set:
        coco_anno_file = open('./road_annotations/coco_annotation_train2_quarter.json', 'w')
    else:
        coco_anno_file = open('./coco_annotation_val2.json', 'w')
    
    p = argparse.ArgumentParser(description='extract frame from videos')
    p.add_argument('--data_dir', type=str,
                   help='Video directory where videos are saved.')
    args = p.parse_args()
    input_images_dir = os.path.join(args.data_dir, 'rgb-images')
    video_dirs = os.listdir(input_images_dir)
    video_dirs = [af for af in video_dirs if len(af)>3]
    print('NUMBER OF VIDEO FILES are:::>', len(video_dirs))

    ## read train and val annotations
    anno_file  = os.path.join(args.data_dir, 'road_trainval_v1.0.json')
    with open(anno_file,'r') as fff:
        final_annots = json.load(fff)

    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []

    # ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL']
    agent_labels = final_annots['agent_labels']
    for idx, label_name in enumerate(agent_labels):
        out_json['categories'].append({'id': idx, 'name': label_name})
    print(out_json['categories'])

    train_video_list = []
    val_video_list = []
    for i in final_annots['db'].keys():
        if 'train_2' in final_annots['db'][i]['split_ids']:
            train_video_list.append(i)
        if 'val_2' in final_annots['db'][i]['split_ids']:
            val_video_list.append(i)

    print('All video list:')
    print(video_dirs)
    print('Train video list:')
    print(train_video_list)
    print('Val video list:')
    print(val_video_list)
    
    if is_train_set:
        video_list = train_video_list
    else:
        video_list = val_video_list

    for video_name in video_list:
        print(' Creating for', video_name, '\n\n')

        label_types = ['agent'] # or = final_annots['label_types']
        tube_uids = {}
        database = final_annots['db'][video_name]
        frames = database['frames']
        frame_nums = [int(f) for f in frames.keys()]
        for frame_num in sorted(frame_nums): #loop from first frame to last
            if (frame_num % 4 == 0):
                frame_id = str(frame_num)
                img_path = input_images_dir + '/{:s}/{:05d}.jpg'.format(video_name, frame_num)
                frame = Image.open(img_path)
                w, h = frame.size

                # check if frame is annotated
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                #if frame_id in frames.keys():
                    if frames[frame_id]['annotated'] == 0:
                        frames[frame_id]['annos'] = {}
                    frame_annos = frames[frame_id]['annos']

                    image_info = dict(
                        file_name = img_path,
                        height = h,
                        width = w,
                        id = img_id
                    )            

                    out_json['images'].append(image_info)

                    for key in frame_annos:
                        anno = frame_annos[key]
                        box = anno['box']
                        box[0] *= w; box[1] *= h; box[2] *= w; box[3] *= h
                        labels = []

                        for idx, label_type in enumerate(label_types):
                            ## Get ids for only the classesbeing used 
                            filtered_ids = filter_labels(anno[label_type+'_ids'], final_annots['all_'+label_type+'_labels'], final_annots[label_type+'_labels'])   
                            classes = final_annots[label_type+'_labels'] ## classes that are being currently used 
                            # all_classes = final_annots['all_'+'agent'+'_labels'] ## All classes of this label type that are annotated
                            for fid in filtered_ids:
                                labels.append(classes[fid])          

                        assert len(labels)==1

                        box[2] = box[2] - box[0]
                        box[3] = box[3] - box[1]
                        area = box[2] * box[3]
                        anno_info = dict(
                            iscrowd = 0,
                            category_id = agent_labels.index(labels[0]),                        
                            bbox = box,
                            area = area,
                            image_id = img_id,
                            id = ann_id
                        )
                        out_json['annotations'].append(anno_info)
                        ann_id += 1
                    img_id += 1
        
print(f'img_id: {img_id}')
print(f'ann_id: {ann_id}')

json.dump(out_json, coco_anno_file)


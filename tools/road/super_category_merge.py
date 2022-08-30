"""
The purpose of this script is to merge certain classes of ROAD dataset into super-categories.

"""

import json

# ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL']
# 0 Ped
# 1 Vehicle
# 2 Cyc
# 3 Mobike
# 4 TL/OthTL
# 5 Inactive

# provide the annotation file
anno_file = './road_annotations/coco_annotation_train2_quarter_inactive.json'

# path to save the new annotation with super-categories
out_file = './road_annotations/coco_annotation_train2_quarter_inactive_merged.json'

with open(anno_file, 'r') as f:
    data = json.load(f)

print(data.keys())
for anno in data['annotations']:
    if anno['category_id'] in [1, 4, 5, 6, 7]:
        anno['category_id'] = 1
    if anno['category_id'] in [8,9]:
        anno['category_id'] = 4
    if anno['category_id'] == 10:
        anno['category_id'] = 5

data['categories'] = [
    {'id': 0, 'name': 'Ped'}, 
    {'id': 1, 'name': 'Vehicle'},
    {'id': 2, 'name': 'Cyc'},
    {'id': 3, 'name': 'Mobike'},
    {'id': 4, 'name': 'TL'},
    {'id': 5, 'name': 'Inactive'}]


json.dump(data, open(out_file,'w'))

# the new config inherits the base configs to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3,5])

runner = dict(type='EpochBasedRunner', max_epochs=7)

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Ped', 'Vehicle', 'Cyc', 'Mobike', 'TL', 'Inactive')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/road/detections/coco_annotation_train1_quarter_inactive_merged.json',
        img_prefix='/road/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/road/detections/coco_annotation_val1_inactive_merged.json',
        img_prefix='/road/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/road/detections/coco_annotation_val1_inactive_merged.json',
        img_prefix='/road/'))

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
# model = dict(
#     roi_head=dict(
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 # explicitly over-write all the `num_classes` field from default 80 to 5.
#                 num_classes=10),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 # explicitly over-write all the `num_classes` field from default 80 to 5.
#                 num_classes=10),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 # explicitly over-write all the `num_classes` field from default 80 to 5.
#                 num_classes=10)]
#                 )
#             )

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=6)))

evaluation=dict(classwise=True, metric='bbox')
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

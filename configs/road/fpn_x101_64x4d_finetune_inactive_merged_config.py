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

model = dict(
    roi_head=dict(
    bbox_head=dict(num_classes=6)),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth"


evaluation=dict(classwise=True, metric='bbox')

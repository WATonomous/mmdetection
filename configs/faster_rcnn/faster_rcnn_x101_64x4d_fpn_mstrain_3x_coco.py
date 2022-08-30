_base_ = [
    '../common/mstrain_3x_coco.py', '../_base_/models/faster_rcnn_r50_fpn.py'
]
model = dict(
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file='./road_annotations/coco_annotation_train2_quarter.json',
        img_prefix='/data/dataset/road-dataset/road/'),
    val=dict(
        ann_file='./road_annotations/coco_annotation_train2_quarter.json',
        img_prefix='/data/dataset/road-dataset/road/'),
    test=dict(
        ann_file='./road_annotations/coco_annotation_train2_quarter.json',
        img_prefix='/data/dataset/road-dataset/road/'))

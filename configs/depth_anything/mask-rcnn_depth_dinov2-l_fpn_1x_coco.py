_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn_no_backbone.py',
    './coco_instance_dvit.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='DINOv2',
        version='large',
        freeze=False,
        load_from='/media/home/dhwang/depth_mmdet/checkpoints/depth_anything_vitl14.pth'
    ),
    neck=dict(in_channels=[1024, 1024, 1024, 1024]))

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

_base_ = [
    '/media/home/dhwang/depth_mmdet/configs/_base_/models/mask-rcnn_r50_fpn.py',
    './lsj-100e_city-instance-dvit.py',
]
custom_imports = dict(imports=['projects.ViTDet.vitdet'])
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (700, 700)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=28, batch_augments=batch_augments),
    backbone=dict(
        _delete_ = True,
        type='DINOv2',
        version='native',
        freeze=False,
    ),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=1024,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=8,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_head=dict(num_classes=8)    
        ))

custom_hooks = [dict(type='Fp16CompresssionHook')]
find_unused_parameters = True
#resume=True

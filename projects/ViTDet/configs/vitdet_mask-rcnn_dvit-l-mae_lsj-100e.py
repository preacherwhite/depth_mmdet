_base_ = [
    '/media/home/dhwang/depth_mmdet/configs/_base_/models/mask-rcnn_r50_fpn_no_backbone.py',
    './lsj-100e_coco-instance.py',
]
custom_imports = dict(imports=['projects.ViTDet.vitdet'])
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1022, 1022)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=28, batch_augments=batch_augments),
    backbone=dict(
        type='DINOv2',
        version='large',
        freeze=False,
        load_from='/media/home/dhwang/depth_mmdet/checkpoints/depth_anything_vitl14.pth'
    ),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

custom_hooks = [dict(type='Fp16CompresssionHook')]
find_unused_parameters = True
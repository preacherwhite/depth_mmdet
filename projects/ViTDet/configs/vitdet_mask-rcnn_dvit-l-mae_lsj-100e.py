_base_ = [
    '/media/home/dhwang/depth_mmdet/configs/_base_/models/mask-rcnn_r50_fpn_no_backbone.py',
    './lsj-100e_coco-instance.py',
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        type='DINOv2',
        version='large',
        freeze=False,
        load_from='/media/home/dhwang/depth_mmdet/checkpoints/depth_anything_vitl14.pth'
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
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

custom_hooks = [dict(type='Fp16CompresssionHook')]

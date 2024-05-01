"""
This file is modified from https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py which is under Apache 2.0 license.
"""

_base_ = ['default_runtime.py']



# runtime
train_cfg = dict(max_epochs=200, val_interval=10)
randomness = dict(seed=21)

base_lr=1e-4
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        milestones=[170],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

# codec settings
# codec = dict(
#     type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         type='HRNet',
#         in_channels=3,
#         extra=dict(
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(4, ),
#                 num_channels=(64, )),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='BASIC',
#                 num_blocks=(4, 4),
#                 num_channels=(32, 64)),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4),
#                 num_channels=(32, 64, 128)),
#             stage4=dict(
#                 num_modules=3,
#                 num_branches=4,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4, 4),
#                 num_channels=(32, 64, 128, 256))),
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='https://download.openmmlab.com/mmpose/'
#             'pretrain_models/hrnet_w32-36af842e.pth'
#             ),
#     ),
#     head=dict(
#         type='HeatmapHead',
#         in_channels=32,
#         out_channels=17,
#         deconv_out_channels=None,
#         loss=dict(type='KeypointMSELoss', use_target_weight=True),
#         decoder=codec),
#     test_cfg=dict(
#         flip_test=True,
#         flip_mode='heatmap',
#         shift_heatmap=True,
#     ))

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/mnt/localssd/fake_clothes/'

# pipelines
# train_pipeline = [
#     dict(type='LoadImage'),
#     dict(type='GetBBoxCenterScale'),
#     dict(type='RandomFlip', direction='horizontal'),
#     dict(type='RandomBBoxTransform'),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='GenerateTarget', encoder=codec),
#     dict(type='PackPoseInputs')
# ]
# val_pipeline = [
#     dict(type='LoadImage'),
#     dict(type='GetBBoxCenterScale'),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='PackPoseInputs')
# ]
backend_args = dict(backend='local')
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    #dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# dataset and dataloader settings
train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        data_root=data_root,
        ann_file='new_train_clothes_anno_lower.json',
        data_prefix=dict(img='lower/'),
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        pipeline=train_pipeline,
        test_mode=False,
        ),
    )

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='new_test_clothes_anno_lower.json',
        data_prefix=dict(img='test_lower/'),
        test_mode=True,
        pipeline=val_pipeline,
        ),
    )

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test_clothes_anno_lower.json')
test_evaluator = val_evaluator

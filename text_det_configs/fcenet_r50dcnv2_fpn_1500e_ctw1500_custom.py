# should be placed under mmocr/configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom.py
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_6e_custom.py',
    '../../_base_/det_models/fcenet_r50dcnv2_fpn.py',
    '../../_base_/det_datasets/comics_speech_bubble_dataset.py',
    '../../_base_/det_pipelines/fcenet_pipeline.py'
]

train_list = {{_base_.train_list}}
val_list = {{_base_.val_list}}
test_list = {{_base_.test_list}}

train_pipeline_ctw1500 = {{_base_.train_pipeline_ctw1500}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

load_from = '/scratch/users/gsoykan20/projects/mmocr/work_dirs/base_models/fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=5,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_ctw1500),
    val=dict(
        type='UniformConcatDataset',
        datasets=val_list,
        pipeline=test_pipeline_ctw1500),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500))

evaluation = dict(
    interval=1,
    metric='hmean-iou',
    save_best='0_hmean-iou:hmean',
    rule='greater')
checkpoint_config = dict(interval=100)  # for saving regardless

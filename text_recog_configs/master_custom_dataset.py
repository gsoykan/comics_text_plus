# should be placed under mmocr/configs/textrecog/master/master_custom_dataset.py

_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_adam_step_6e_custom.py',
    '../../_base_/recog_datasets/comic_speech_bubble_dataset.py',
    '../../_base_/recog_models/master.py',
    '../../_base_/recog_pipelines/master_pipeline.py',
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
val_list = {{_base_.val_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

load_from = '/scratch/users/gsoykan20/projects/mmocr/work_dirs/base_models/master_r31_12e_ST_MJ_SA-787edd36.pth'

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=5,
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=val_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1,
    metric="acc",
    save_best="0_1-N.E.D", #"0_char_precision",
    rule="greater"
)  # for best saving
checkpoint_config = dict(interval=100)

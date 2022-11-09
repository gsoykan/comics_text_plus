# should be placed under mmocr/configs/_base_/schedules/schedule_adam_step_6e_custom.py

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=0.5))
# learning policy
lr_config = dict(policy='step', step=[3, 4])
# lr_config = dict(
#    policy='step',
#    step=[3, 4],
#    warmup='linear',
#    warmup_iters=1,
#    warmup_ratio=0.001,
#    warmup_by_epoch=True)

total_epochs = 6

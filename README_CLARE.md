#### teleoperation
```
python crisp_gym/scripts/record_lerobot_format_leader_follower.py --repo-id=continuallearning/{repo_id} --tasks={task_name} --resume
```


#### clean dataset
```
python scripts/dataset_conversions/filter_zero_actions.py --source=continuallearning/{original_repo_id}  --target=continuallearning/{target_repo_id} --push
```

#### deploy
```
python crisp_gym/scripts/deploy_policy.py --repo-id=continuallearning/{repo_id} --push-to-hub --path continuallearning/{model_id} --policy-config 0_default_config
```


python crisp_gym/scripts/deploy_policy.py --repo-id=continuallearning/eval_test --path continuallearning/diffusion_fft_real_0_put_bowl_filtered --policy-config 0_default_config

python crisp_gym/scripts/deploy_policy.py --repo-id=continuallearning/eval_dit_fft_real_0_put_bowl_filtered_seed1000 --path continuallearning/dit_fft_real_0_put_bowl_filtered_seed1000 --policy-config 0_default_config

python crisp_gym/scripts/deploy_policy.py --repo-id=continuallearning/eval_dit_posttrain_real_0_put_bowl_filtered_seed1000 --path continuallearning/dit_posttrain_real_0_put_bowl_filtered_seed1000 --policy-config 0_default_config

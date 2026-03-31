from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="continuallearning/eval_dit_flow_mt_cl_seed_42_franka_task_1_stack_bowls")
dataset.push_to_hub(private=False)
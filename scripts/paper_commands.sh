# train a detector for each environment
python scripts/train_detector.py --img_scale 4 --grayscale True --save_dir ~/models --prog_bar False --upload_model True --config_path assets/configs/ACC.toml --devices 0 &
python scripts/train_detector.py --img_scale 4 --grayscale True --save_dir ~/models --prog_bar False --upload_model True --config_path assets/configs/Pointmesses.toml --devices 1 &
python scripts/train_detector.py --img_scale 4 --grayscale True --save_dir ~/models --prog_bar False --upload_model True --config_path assets/configs/PMGoalFinding.toml --devices 2 &

# train an RL agent on each environment

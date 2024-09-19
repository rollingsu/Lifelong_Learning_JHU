$env:ckpt = "D:/SegVol-main/work_dir/20240807-1220/medsam_model_e40.pth"
$env:work_dir = "./work_dir"
$env:demo_config_path = "./config/config_demo.json"

python inference_demo.py `
--resume $env:ckpt `
-work_dir $env:work_dir `
--demo_config $env:demo_config_path
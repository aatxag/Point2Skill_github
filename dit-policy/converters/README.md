# Convertidoreak erabiltzeko
cd /home/labiiwa/Point2Skill_github/dit-policy/converters

python3 convert_to_robobuf_contact_hindsight.py \
  --dataset_dir /home/labiiwa/dit_demos/close \
  --out_path /home/labiiwa/dit-policy/bc_finetune/close/close_dataset.rds \
  --hand_eye_yaml my_data/hand_eye_result.yaml \
  --trim_start_motion \
  --dedup_static


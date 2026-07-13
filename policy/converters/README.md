# Convertidoreak erabiltzeko
cd /home/labiiwa/Point2Skill_github/dit-policy/converters

python3 convert_to_robobuf_contact_hindsight.py \
  --dataset_dir /home/labiiwa/dit_demos/pick_experiment \
  --out_path /home/labiiwa/dit-policy/my_data/pick_experiment_200 \
  --hand_eye_yaml my_data/hand_eye_result.yaml \
  --trim_start_motion \
  --dedup_static



python3 convert_to_robobuf_place_hindsight.py \
  --dataset_dir /home/labiiwa/dit_demos/place_experiment \
  --out_path /home/labiiwa/dit-policy/my_data/place_experiment_200 \
  --hand_eye_yaml my_data/hand_eye_result.yaml \
  --trim_start_motion \
  --dedup_static


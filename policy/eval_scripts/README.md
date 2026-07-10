# Eval Scripts
## FUNCTION
- Cargar el modelo
- CArgar los pesos
- CArgas las normalizaciones
- Conectar con el robot y las camaras
- Mover el robot al principio
### CLASE DEL PUNTO
- Inicializar el interface para elegir punto de contacto
Una vez se tiene el punto
ANTES DEL GRASP
1. Siempre primero llevarlo a base para tener de referencia
2. Encada paso, rollout, este punto se actualiza teniendo en cuenta la pose del robot y el hand-eye
DESPUES DEL GRASP
  - Si no hay depth se mira en el venicdario, si hay primero de todo se normaliza.
1. Se congela el punto respecto al end effector
### CLASE DE POLICY
- El objetivo es cargar el modelo
- Controla el procesamiento de imagenes
- Action chunk
- Policy forward
  - suavizar con gamma
  - lift scale
  -    
## How to use

- Open politikarako, armairuarekin default posizioa:
python3 eval_franka_2cam_contact.py   /home/labiiwa/Point2Skill_github/dit-policy/bc_finetune/open/wandb_None_franka_2cam_contact_resnet_gn_2026-06-09_14-25-12/open.ckpt   --gamma 1.0 --T 500 --action_idx 3 --lift_scale 2.0 

- Pick coffee: 
python3 eval_franka_2cam_contact.py /home/labiiwa/dit-policy/bc_finetune/pick_coffee/wandb_None_franka_2cam_contact_resnet_gn_2026-06-10_10-53-01/pick_coffee.ckpt   --gamma 1.0   --T 500   --action_idx 3   --auto_lift   --grasp_confirm_steps 3   --lift_trigger 0.04   --q_start -0.29019999504089355 -0.5712000131607056 0.15060000121593475 -2.533400058746338 0.11640000343322754 1.8801000118255615 -2.587100028991699


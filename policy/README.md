
├── policy/
│   ├── finetune.py              # Versión original (sin obs_config.yaml ni wandb.log).
│   ├── setup.py, env.yml        # Paquete data4robotics + entorno conda.
│   ├── jobs.sh, diffuse_jobs.sh # Lanzadores antiguos estilo slurm/bash.
│   │
│   ├── converters/
│   │   ├── convert_to_robobuf_contact.py            # versión base
│   │   ├── convert_to_robobuf_contact_hindsight.py  # ★ la principal (980 líneas)
│   │   ├── convert_to_robobuf_place_hindsight.py    # variante Place
│   │   └── my_data/camera_intrinsics.yaml, hand_eye_result.yaml  # fx,fy,cx,cy y T_cam_to_ee
│   │
│   ├── data4robotics/
│   │   ├── __init__.py          # solo exporta load_resnet18, load_vit
│   │   ├── agent.py             # BaseAgent: tokenize_obs (imgs→tokens + obs como token extra)
│   │   ├── load_pretrained.py   # carga ResNet18/ViT preentrenados
│   │   ├── models/
│   │   │   ├── diffusion_contact.py   # ★ DiffusionTransformerAgent + _DiTNoiseNet
│   │   │   ├── diffusion_unet.py      # variante UNet
│   │   │   ├── resnet.py, vit.py      # encoders visuales
│   │   │   ├── base.py, action_distributions.py, action_transformer.py
│   │   ├── replay_buffer_contact.py   # ★ RobobufReplayBuffer (batch de 4 elementos)
│   │   ├── task_contact.py            # ★ BCTaskContact (eval: val loss, L2, LSig)
│   │   │                              # ⚠ importa data4robotics.task, que NO está en el
│   │   │                              #   source tree (solo en install/, ver §8)
│   │   ├── trainers/
│   │   │   ├── base.py                # BaseTrainer: optim, scheduler, checkpoints, log
│   │   │   ├── bc_contact.py          # ★ BehaviorCloning.training_step (desempaqueta 3 o 4)
│   │   │   └── utils.py               # optim_builder, schedule_builder
│   │   ├── transforms.py              # preproc / medium / gpu transforms
│   │   └── misc.py                    # init_job, GLOBAL_STEP, checkpoint handler
│   │
│   ├── experiments/                   # configs Hydra
│   │   ├── finetune_contact.yaml      # defaults: diffusion_contact + franka_2cam_contact + bc_contact
│   │   ├── finetune.yaml              # versión sin contacto
│   │   ├── agent/diffusion_contact.yaml       # DiT: hidden 512, 6 bloques, 8 heads,
│   │   │                                      # train_steps=100, eval_steps=8 (DDIM)
│   │   ├── agent/features/resnet_gn_nopool.yaml  # encoder por defecto del agente contact
│   │   ├── task/franka_2cam_contact.yaml      # obs_dim=8, ac_dim=8, cams [0,1]
│   │   ├── trainer/bc_contact.yaml            # AdamW + cosine warmup 2000
│   │   └── hydra/launcher/slurm.yaml
│   │
│   ├── eval_scripts/
│   │   ├── eval_franka_2cam_contact.py        # ★ eval con click humano (832 líneas)
│   │   ├── eval_franka_env_2cam_contact.py    # make_fr3_env_2cam_contact (env ROS2)
│   │   ├── eval_franka_2cam_contact_position.py  # variante: recibe centroide del VLM
│   │   ├── eval_franka_2cam_place_contact.py / eval_franka_2cam_contact_place.py  # Place
│   │   └── README.md
│   │
│   └── install/                       # ⚠ artefactos de colcon build commiteados.
│       └── .../site-packages/data4robotics/   # aquí SÍ están task.py y replay_buffer.py
│                                              # que faltan en el source tree

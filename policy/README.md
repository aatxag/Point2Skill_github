## ESQUEMA GENERAL

```text
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
```
## FUNCIONAMIENTO

## Grafo de llamadas del entrenamiento 

```text
run_training.py
    │  construye: python3 finetune_contact.py  (⚠ no está en GitHub; en tu máquina
    │             equivale a finetune.py con overrides agent=diffusion_contact,
    │             task=franka_2cam_contact, trainer=bc_contact)
    ↓
finetune.py :: bc_finetune(cfg)              [@hydra.main → experiments/finetune*.yaml]
    │
    ├── misc.init_job(cfg)                   → wandb init, resume detection
    ├── escribe agent_config.yaml            → lo reutiliza el eval script
    ├── escribe obs_config.yaml              → cams + transform "preproc" para eval
    │
    ├── agent   = hydra.utils.instantiate(cfg.agent)
    │       → data4robotics.models.diffusion_contact.DiffusionTransformerAgent
    │           ├── BaseAgent.__init__       (agent.py: encoder visual + obs token)
    │           └── _DiTNoiseNet(...)        (noise net con contact_emb)
    │
    ├── trainer = hydra.utils.instantiate(cfg.trainer, model=agent)
    │       → data4robotics.trainers.bc_contact.BehaviorCloning (hereda BaseTrainer)
    │
    ├── task    = hydra.utils.instantiate(cfg.task, batch_size, num_workers)
    │       → data4robotics.task_contact.BCTaskContact (hereda DefaultTask)
    │           ├── train_buffer → replay_buffer_contact.RobobufReplayBuffer(mode=train)
    │           └── test_buffer  → RobobufReplayBuffer(mode=test)
    │
    └── loop (max_iterations):
            batch = next(task.train_loader)
            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, GLOBAL_STEP)
                 └── (imgs,obs), actions, mask, contact_point = batch   # si len==4
                     model(imgs, obs, ac_flat, mask_flat, contact_point=contact_point)
            loss.backward(); trainer.optim.step()
            cada schedule_freq → trainer.step_schedule()
            cada eval_freq     → task.eval(trainer, step)   # val loss + AC L2 + AC LSig
            cada save_freq     → trainer.save_checkpoint()
```

Cómo llega el `contact_point` al modelo, extremo a extremo:

```text
buf.pkl: t.obs.obs["contact_anchor"]   (ya normalizado por el converter)
    ↓  RobobufReplayBuffer.__init__ lo extrae por transición
    ↓  __getitem__ devuelve ((imgs,obs), a_t, mask, contact_anchor)  ← batch de 4
    ↓  BehaviorCloning.training_step lo desempaqueta y lo pasa como kwarg
    ↓  DiffusionTransformerAgent.forward(..., contact_point=...)
    ↓  _DiTNoiseNet.forward_dec:
           time_enc = time_net(τ)
           if contact_point: time_enc += contact_emb(contact_point)   ← adaLN-zero
    ↓  time_enc modula TODOS los bloques _DiTDecoder vía _ShiftScaleMod/_ZeroScaleMod
       (cond = mean(enc_cache) + time_enc)  y también _FinalLayer
```

Detalle importante del adaLN-zero: la última capa de `contact_emb` está inicializada a cero (pesos y bias), así que al inicio el contacto no perturba los pesos preentrenados — exactamente como lo describes en el paper.

---

## Pipeline de datos (converters)

`convert_to_robobuf_contact_hindsight.py` es el converter principal:

```text
episodios crudos (rgb cam0/cam1, cam0_depth, ee_pose, gripper, gripper_cmd)
    ↓
detect_contact_frame(g_trace, cmd_trace)
    # 1. primer 1→-1 en gripper_cmd (cmd_close)
    # 2. valida caída física ≥ min_drop
    # 3. frame donde la apertura se estabiliza (plateau) = contacto físico
    ↓
get_contact_anchor(depth, frame_grasp, ee_pose, T_cam_to_ee, intrínsecos, u, v)
    # píxel (u,v): CONTACT_U/V por defecto, o contact_pixel.json por episodio
    # depth inválido → find_nearest_valid_depth_pixel (radio DEPTH_SEARCH_RADIUS)
    # backproject (u,v,depth) → p_cam
    # p_ee   = T_cam_to_ee @ p_cam
    # p_base = T_ee_to_base @ T_cam_to_ee @ p_cam
    ↓
hindsight relabeling: anchor re-expresado en el frame EE de cada paso
    # pre-grasp: EE_grasp → base → EE_i (el vector se encoge hacia 0)
    # ventana ± (contact_keep_before=3, contact_keep_after=8) preservada en dedup
    ↓
normalización global del dataset:
    contact_loc   = (c_min + c_max)/2
    contact_scale = (c_max − c_min)/2  (clip min 1e-6)
    ca_norm = clip((anchor − loc)/scale, −1, 1)
    ↓
salidas:
    buf.pkl            (obs.obs["contact_anchor"] YA normalizado)
    ac_norm.json       (loc/scale de acciones)
    contact_norm.json  ({"loc": [...], "scale": [...]})  ← claves loc/scale, no mean/std
    contact_debug.json (píxel usado, fallback sí/no, depth, p_cam/p_ee/p_base por episodio)
```

Nota de coherencia: el replay buffer NO re-normaliza — el comentario en `__getitem__` lo dice explícitamente ("already normalized by the converter"). `contact_norm.json` se carga solo para (a) copiarlo junto al checkpoint y (b) decidir si el conditioning está activo (`_contact_loc is not None`).

---


## Structure

```text
└── vlm_service/                       # workspace ROS2 (3 paquetes)
    ├── run_planner.sh
    ├── service_language/              # nodo de entrada de lenguaje
    │   └── src/service_language/language_publisher.py (+ _answer.py)
    │       # publica el comando del usuario en /topic_prompt
    ├── service_planner/               # ★ el cerebro VLM
    │   └── src/service_planner/planner.py (740 líneas)
    │       # Qwen-VL genera plan JSON → para PICK: bbox Qwen → máscara SAM (vit_b)
    │       # → centroide → /perception/centroid; pasos → /selected_policy
    └── service_primitives/            # ★ ejecutor de primitivas
        └── src/service_primitives/primitives.py
            # MODELS: dict primitiva → {script eval, ckpt, args, q_start, home_q}
            # escucha /selected_policy, lanza el eval script como subproceso,
            # publica /policy_execution_status, gestiona stop de emergencia
```text
1. LANGUAGE

source  ~/Point2Skill_github/vlm_service/install/setup.bash
ros2 launch service_language language.launch.py

2. PLANNERRA

source /home/labiiwa/ros2_ws/install/setup.bash
ros2 launch service_planner planner.launch.py

3. PRIMITIBAK

source ~/.bashrc
ros2 launch service_primitives primitives.launch.py




# Language en sarrera

" Put the coffee inside the drawer "

Bere plana: 
```text
[python3-1] [INFO] [1781188975.043684057] [service_planner]: [Planner] Command: 'Put the coffee inside the drawer'
[python3-1] [INFO] [1781188975.045223532] [service_planner]: [Planner] → Generating plan…
[python3-1] The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[python3-1] [INFO] [1781188978.870979723] [service_planner]: [Planner] Raw plan output:
[python3-1] [{"step":1,"action":"open","object":null,"target":"drawer"},{"step":2,"action":"pick","object":"coffee","target":null},{"step":3,"action":"place","object":"coffee","target":"drawer"}]
[python3-1] [INFO] [1781188978.871547449] [service_planner]: [Planner] Keyword match → open
[python3-1] [INFO] [1781188979.682878163] [service_planner]: [Planner] VLM policy answer: 'pick_coffee'
[python3-1] [INFO] [1781188979.683191566] [service_planner]: [Planner] VLM match → pick_coffee
[python3-1] [INFO] [1781188979.683498338] [service_planner]: [Planner] Keyword match → place_drawer
[python3-1] [INFO] [1781188979.683761477] [service_planner]: [Planner]
[python3-1] Execution plan:
[python3-1]   1. OPEN  → drawer  [policy: open]
[python3-1]   2. PICK coffee  [policy: pick_coffee]
[python3-1]   3. PLACE coffee → drawer  [policy: place_drawer]
[python3-1] [INFO] [1781188979.684047756] [service_planner]: [Planner] → Execution plan:
[python3-1]   1. OPEN  → drawer  [policy: open]
[python3-1]   2. PICK coffee  [policy: pick_coffee]
[python3-1]   3. PLACE coffee → drawer  [policy: place_drawer]

```text

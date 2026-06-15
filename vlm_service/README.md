1. PRIMITIBAK

source ~/Point2Skill_github/vlm_service/install/setup.bash
ros2 launch service_primitives primitives.launch.py

2. PLANNERRA

source ~/Point2Skill_github/vlm_service/install/setup.bash
ros2 launch service_planner planner.launch.py

3. LANGUAGE

source  ~/Point2Skill_github/vlm_service/install/setup.bash
ros2 launch service_language language.launch.py



# Language en sarrera

" Put the coffee inside the drawer "

Bere plana: 

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


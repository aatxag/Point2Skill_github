## SCRIPTS

replay_buffer_contact.py

- Defines dataset of pytorch from robobuf data to train the policy.

   buf.pkl

  ↓

Carga trayectorias de robot
   
   ↓

Separa train / test
   
   ↓

Para cada transición extrae:
   - imágenes de cámaras
   - estado del robot
   - acciones futuras
   - máscara de acciones válidas
   - contact anchor opcional
   
   ↓

Devuelve tensores de PyTorch

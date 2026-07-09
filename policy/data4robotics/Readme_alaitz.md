## SCRIPTS

### replay_buffer_contact.py

- Defines dataset of pytorch from robobuf data to train the policy.
     - buf.pkl
     - Carga trayectorias de robot
     - Separa train / test
     - Para cada transición extrae:
        - imágenes de cámaras
      - estado del robot
      - acciones futuras
      - máscara de acciones válidas
      - contact anchor opcional
- Logica completa:
1. Carga el replay buffer.

2. Carga ac_norm.json si existe.

3. Carga contact_norm.json si existe.

4. Hace shuffle reproducible.

5. Divide train/test.

6. Para cada transición:
      - obtiene una secuencia de acciones futuras;
      - crea una máscara;
      - hace padding si llega al final;
      - extrae el contact_anchor.

7. Cuando se solicita una muestra:
      - obtiene las imágenes;
      - añade frames anteriores;
      - opcionalmente añade goal image;
      - convierte todo a tensores;
      - devuelve observaciones, acciones, máscara
        y posiblemente contact_anchor.

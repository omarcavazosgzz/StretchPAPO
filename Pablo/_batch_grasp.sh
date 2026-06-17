#!/bin/bash
# Corre varios agarres (objeto + spawn) y resume el resultado de cada uno.
cd /home/alumno1234/Documents/Ubuntus/STRECH2/stretch_mujoco_digital_twin || exit 1
run() {  # $1=obj  $2=spawn
  local obj="$1" spawn="$2"
  local out
  out=$(STRETCH_FIXED_SPAWN="$spawn" timeout 380 env MUJOCO_GL=egl .venv/bin/python Pablo/fase2.py "$obj" 2>&1 \
        | grep -E "objeto z|AGARRE OK|REVISAR|pre-cierre")
  echo "===== obj=$obj spawn=[$spawn] ====="
  echo "$out" | tail -3
}
run cubo_rojo "2.25,-0.8,90"
run cubo_rojo "2.45,-0.95,90"
run tomate    "2.25,-0.8,90"
run huevo     "2.25,-0.8,90"
echo "ALL DONE"

#!/bin/bash
set -euxo pipefail
export TMPDIR=$HOME/tmp
rm -rf $HOME/fplus-venv
python3 -m venv $HOME/fplus-venv
. $HOME/fplus-venv/bin/activate
pip install --upgrade pip setuptools wheel
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation egl_probe hf-egl-probe
cd $HOME/lerobot
pip install -e ".[fastwam,libero]"
pip install "robosuite==1.4.1" "bddl==1.0.1" "easydict==1.13" "mujoco==3.7.0" "matplotlib==3.10.8" "Wand==0.6.13" "scikit-image==0.25.2" "gym==0.26.2"
pip install --no-deps -e $HOME/libero-plus
pip uninstall -y hf-libero
pip list | grep -iE "^(lerobot|robosuite|libero|wand|mujoco|numpy) "
echo INSTALL-DONE

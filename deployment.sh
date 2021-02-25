#!/bin/bash

python prepare_for_deployment.py NonParametric/model_arch.h5 NonParametric/model.pb
python prepare_for_deployment.py NonParametric/model_arch.h5 NonParametric/bestModel/model.pb
#python prepare_for_deployment.py Parametric/model_arch.h5 Parametric/model.pb
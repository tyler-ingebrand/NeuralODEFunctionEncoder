#!/bin/bash

python src/Predictors/MLP.py
python src/Predictors/NeuralODE.py

python src/Predictors/FE.py
python src/Predictors/FE_NeuralODE.py

python src/Predictors/FE_Residuals.py
python src/Predictors/FE_NeuralODE_Residuals.py


echo "All tests passed"

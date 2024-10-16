# Zero-Shot Transfer of Neural ODEs

See the [project page](https://tyler-ingebrand.github.io/NeuralODEFunctionEncoder/).

This repository contains the code for the paper "Zero-Shot Transfer of Neural ODEs".
All experiments are run using "./run_experiment.sh" script. 
We expect it to take around 1-2 weeks for all experiments, as there are numerous
baselines and numerous seeds, each of which can take a few hours. 

The main required packages are torch, numpy, gymnasium[mujoco], and safe-control-gym. 


The following are the dependencies for the environment in the experiments, although this repo should not be too sensitive to exact versions:
```
Package                    Version
-------------------------- -----------
absl-py                    2.1.0
ale-py                     0.8.1
AutoROM                    0.6.1
AutoROM.accept-rom-license 0.6.1
casadi                     3.6.4
certifi                    2024.2.2
cffi                       1.16.0
cfgv                       3.4.0
charset-normalizer         3.3.2
clarabel                   0.7.0
click                      8.1.7
cloudpickle                3.0.0
colorednoise               2.2.0
contourpy                  1.2.0
cvxpy                      1.4.2
cycler                     0.12.1
Cython                     3.0.9
dict-deep                  4.1.2
distlib                    0.3.8
distro                     1.9.0
ecos                       2.0.13
etils                      1.7.0
exceptiongroup             1.2.0
Farama-Notifications       0.0.4
fasteners                  0.19
filelock                   3.13.1
fonttools                  4.49.0
fsspec                     2024.2.0
glfw                       2.7.0
gpytorch                   1.11
grpcio                     1.62.0
gymnasium                  0.28.1
identify                   2.5.35
idna                       3.6
imageio                    2.34.0
importlib_resources        6.1.2
iniconfig                  2.0.0
jax-jumpy                  1.0.0
jaxtyping                  0.2.25
Jinja2                     3.1.3
joblib                     1.3.2
kiwisolver                 1.4.5
linear-operator            0.5.2
Markdown                   3.5.2
markdown-it-py             3.0.0
MarkupSafe                 2.1.5
matplotlib                 3.8.3
mdurl                      0.1.2
Mosek                      10.1.27
mpmath                     1.3.0
mujoco                     3.1.3
mujoco-py                  2.1.2.14
munch                      2.5.0
networkx                   3.2.1
nodeenv                    1.8.0
numpy                      1.26.4
nvidia-cublas-cu11         11.10.3.66
nvidia-cublas-cu12         12.1.3.1
nvidia-cuda-cupti-cu12     12.1.105
nvidia-cuda-nvrtc-cu11     11.7.99
nvidia-cuda-nvrtc-cu12     12.1.105
nvidia-cuda-runtime-cu11   11.7.99
nvidia-cuda-runtime-cu12   12.1.105
nvidia-cudnn-cu11          8.5.0.96
nvidia-cudnn-cu12          8.9.2.26
nvidia-cufft-cu12          11.0.2.54
nvidia-curand-cu12         10.3.2.106
nvidia-cusolver-cu12       11.4.5.107
nvidia-cusparse-cu12       12.1.0.106
nvidia-nccl-cu12           2.19.3
nvidia-nvjitlink-cu12      12.4.99
nvidia-nvtx-cu12           12.1.105
opencv-python-headless     4.9.0.80
osqp                       0.6.5
packaging                  23.2
pandas                     2.2.1
pillow                     10.2.0
pip                        24.0
platformdirs               4.2.0
pluggy                     1.4.0
pre-commit                 3.6.2
protobuf                   4.25.3
psutil                     5.9.8
pyaml                      23.12.0
pybind11                   2.11.1
pybullet                   3.2.6
pycddlib                   2.1.7
pycparser                  2.21
pygame                     2.5.2
Pygments                   2.17.2
PyOpenGL                   3.1.7
pyparsing                  3.1.1
PyQt5                      5.15.10
PyQt5-Qt5                  5.15.2
PyQt5-sip                  12.13.0
pytest                     7.4.4
python-dateutil            2.9.0
pytope                     0.0.4
pytz                       2024.1
PyYAML                     6.0.1
qdldl                      0.1.7.post0
requests                   2.31.0
rich                       13.7.1
safe-control-gym           2.0.0
scikit-build               0.17.6
scikit-learn               1.4.1.post1
scikit-optimize            0.9.0
scipy                      1.11.4
scs                        3.2.4.post1
setuptools                 69.1.1
Shimmy                     1.3.0
six                        1.16.0
stable-baselines3          2.2.1
sympy                      1.12
tensorboard                2.16.2
tensorboard-data-server    0.7.2
termcolor                  1.1.0
threadpoolctl              3.3.0
tomli                      2.0.1
torch                      2.2.1
tqdm                       4.66.2
triton                     2.2.0
typeguard                  2.13.3
typing_extensions          4.10.0
tzdata                     2024.1
urllib3                    2.2.1
virtualenv                 20.25.1
Werkzeug                   3.0.1
wheel                      0.42.0
zipp                       3.18.1
```

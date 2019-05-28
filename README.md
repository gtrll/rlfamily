# rlfamily #

Code for reproducing the results in the paper: Predictor-Corrector Policy Optimization. Ching-An Cheng, Xinyan Yan, Nathan Ratliff, Byron Boots. ICML 2019.

### Installation ###
Tested in Ubuntu 16.04 and Ubuntu 18.04 with python 3.5, 3.6.

#### Install rlfamily and most of the requirements ####
Preare python3 virtual environment:
```
sudo apt-get install python3-pip
sudo pip3 install virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install --upgrade -r requirements.txt
```
Install this repo and requirements:
```
git clone https://github.com/gtrll/rlfamily.git
pip install --upgrade -r requirements.txt
```
You may need to run
```
export PYTHONPATH="{PYTHONPATH}:[the parent folder of rlfamily repo]"
```


#### Install PyDart2 ####
Installing PyDart2 through pip does not work. So we install it manually.

Install requirements:
```
sudo apt-add-repository ppa:dartsim
sudo apt-get update
sudo apt-get install libdart6-all-dev
sudo apt-get install swig python3-pip python3-pyqt4 python3-pyqt4.qtopengl
```
Some of the requirement installation may incur errors in Ubuntu 18.04. These can likely be ignored. 
Download, compile, and install
```
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python setup.py build build_ext
python setup.py develop
```


#### Install DartEnv ####
This is a slightly modified version of [DartEnv](https://github.com/DartEnv/dart-env). The changes include:

* Make nodisplay as default.
* Add a state property for convenience.

To install it, 
```
git clone https://github.com/gtrll/dartenv.git
cd dartenv
git checkout nodisplay
pip install -e .[dart]
```

#### Troubleshooting ####
If you encounter the error:
```
ModuleNotFoundError: No module named 'pydart2._pydart2_api'

ImportError: libdart.so.6.5: cannot open shared object file: No such file or directory
```
try installing PyDart2 again.



### Reproduce the results in the paper  ###

#### Run experiments ####
Firstly, go to the main folder.
```
cd rlfamily
```

Run experiments for env [env], using first-order oracle [oracle], using base algorithm [alg]:
```
python scripts/batch_run.py [env] configs_[env] -r [oracle] -a [alg]
```

* [env] can be `cp`, `hopper`, `snake`, and `walker3d`.
* [oracle] can be `mf` (Base Algorithm), `lazy1` (last), `agg1` (reply), `sim1` (TrueDyn), 
`sim1_models` (BiasedDyn), `dyna_adv` (Dyna-Adversarial), `adv` (PicCoLO-Adversarial), 
`sim1_opt` (BiasedDyn0.2-FP),
where the names in the parentheses correspond to the ones in the paper.
* [alg] can be `adam`, `natgrad`, `trpo`.

Note that [oracle] and [alg] can be lists. For example:
```
python scripts/batch_run.py hopper configs_hopper -r mf lazy1 agg1 sim1 sim1_models -a adam trpo natgrad
```

#### Plot results ####
Change the name of the folders based on `icml_piccolo_final_configs` in `scripts/plot_configs.py`.
And plot the results using `scripts/plot_icml_piccolo_final.sh`.


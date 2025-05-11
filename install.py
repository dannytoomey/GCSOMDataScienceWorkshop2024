import importlib, subprocess, sys

# Note: At time of commit, Tensorflow in unable to install
# on Python 3.13+. Install on Python 3.12 to run Tensorflow. 
# PyTorch can be installed on 3.13+ 

dependencies=[
    'torch >= 2.7.0',
    'numpy >= 1.20.0',
    'pandas >= 2.0.0',
    'scikit-learn >= 1.0.0',
    'scipy >= 1.5.0',
    'matplotlib >= 3.7.0',
    'tensorflow >= 2.18.0',
    'scikit-learn >= 1.5.0'
]

for mod in dependencies:
    try:
        importlib.import_module(mod)
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", mod])

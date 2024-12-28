import importlib, subprocess, sys

dependencies=[
    'numpy >= 1.20.0',
    'pandas >= 2.0.0',
    'scikit-learn >= 1.0.0',
    'scipy >= 1.5.0'
]

for mod in dependencies:
    try:
        importlib.import_module(mod)
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", mod])

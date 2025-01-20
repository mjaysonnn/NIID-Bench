conda create -n old_env python=3.7 -y


conda activate old_env

pip install -r requirements.txt --no-input
pip install --upgrade torchvision --no-input

python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import torch; print('torch:', torch.__version__)"


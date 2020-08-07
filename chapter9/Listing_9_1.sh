# Code block 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Code block 2: uncompress NEU data
%%shell
ls /content/drive/'My Drive'/NEU-DET.zip
unzip /content/drive/'My Drive'/NEU-DET.zip

# Code block 3: Clone github repository of Tensorflow model project
!git clone https://github.com/ansarisam/models.git

# Code block 4: Install Google protobuf compiler and other dependencies
!sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

# Code block 4: Install dependencies
%%shell
cd models/research
pwd
protoc object_detection/protos/*.proto --python_out=.
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib

# Code block 5: Build models project
%%shell
export PYTHONPATH=$PYTHONPATH:/content/models/research:/content/models/research/slim
cd /content/models/research
python setup.py build
python setup.py install

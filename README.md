# Iris-Recognition-PyTorch
An end-to-end Iris Recognition using PyTorch.


## Installation

* Prepare tools for setup virtual environment (If you have already done, skip it):
```
sudo apt-get install -y python-pip python3-pip cmake
mkdir ~/.virtualenvs
cd ~/.virtualenvs
sudo pip install virtualenv virtualenvwrapper
sudo pip3 install virtualenv virtualenvwrapper
echo "# virtualenv and virtualenvwrapper" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
```

* Create a new virtual environment, named *iris*:
```
virtualenv -p python3.6 iris
workon iris
```

* Clone git and install required packages:
```
git clone --recursive https://github.com/AntiAegis/Iris-Recognition-PyTorch.git
cd Iris-Recognition-PyTorch
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
```

* In this repository, I adapt off-the-shelf models from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), to install it:
```
pip install -e models/pytorch-image-models
```


## Training
* To start the training process, use the command:
```
python train.py --config config/efficientnet_b0.json --device 0
```


## Results

* The model is [EfficientNet-b0](https://arxiv.org/abs/1905.11946), trained by optimizer [SGDR](https://arxiv.org/abs/1608.03983) with 300 epochs: [training config](https://github.com/AntiAegis/Iris-Recognition-PyTorch/blob/master/config/efficientnet_b0.json).


* Loss and accuracy is summarized and plotted as follows:
  
|       | Loss   | Accuracy |
|-------|--------|----------|
| Train | 0.0105 | 1.0000   |
| Valid | 0.0288 | 0.9980   |

<p align="center">
  <img src="https://github.com/AntiAegis/Iris-Recognition-PyTorch/blob/master/pics/loss.png" width="430" alt="accessibility text">
  <img src="https://github.com/AntiAegis/Iris-Recognition-PyTorch/blob/master/pics/acc.png" width="430" alt="accessibility text">
</p>

* To ensure the trained model focuses on iris region inside images, I use [Grad-CAM](https://arxiv.org/abs/1610.02391) to visualize attention of the last feature layer (right before Global Average Pooling). To visualize heatmap, use this command:
```
python visualize.py --image /home/antiaegis/datasets/Iris/MMU2/010105.bmp \
                    --config config/efficientnet_b0.json \
                    --weight /home/antiaegis/checkpoints/model_best.pth \
                    --use-cuda
```

<p align="center">
  <img src="https://github.com/AntiAegis/Iris-Recognition-PyTorch/blob/master/pics/grad_cam.jpg" width="500" alt="accessibility text">
</p>

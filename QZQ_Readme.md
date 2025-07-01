# 下载对应的anaconda
https://www.anaconda.com/download/success 下载minconda即可
## 如果是windows系统，先安装wsl，然后在vscode里打开wsl
从外面下载anaconda的linux版本sh下载后，放到对应folder，然后bash安装
## 如果是mac/linux系统
先安装好对应anaconda

# 在conda环境里创建python 3.11的版本因为coeditor要求python 3.11
conda create -n qzq_coeditor python=3.11
conda activate qzq_coeditor

# 在上述conda环境里安装对应的python包

以pandas为例执行pip install pandas
在安装前可以执行which python或者python
然后import sys
sys.executable()
确定当前是对应python环境
然后退出后执行pip install

pandas

nltk

transformers==4.27.4 (如果装不上，如果你只是data部分就不要指定版本)

torch

dateparser

jedi==0.18.2

ipython

parso

termcolor

editdistance==0.6.2

accelerate>=0.26.0

# 数据的同学先修改notebooks/download_data.ipynb
对于notebook选择对应kernel
先研究这个notebook，确保可以下载自己对应语言的repo，最后会存放到datasets_root下

# 下载完后，为了开发，建议train,test,valid只各自放1个repo，然后从src/prepare_data.py出发研究如何平替jedi

# 自己语言的package如果要安装的话，安装流程，也要把它加到这个readme里面

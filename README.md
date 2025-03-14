# LHGNN：潜在异构图上的链接预测

我们提供了代码（基于 PyTorch）和数据集，用于我们的论文："[潜在异构图上的链接预测](https://arxiv.org/abs/2302.10432)"（简称为 LHGNN），该论文已被 TheWebConf 2023 接收。


## 1. 描述
仓库的组织结构如下：

* dataset/：包含 3 个基准数据集：fb15k-237、wn18rr 和 dblp（稍后我们将上传更大的数据集 ogb-mag）。所有数据集都会在运行时动态处理。请在运行之前解压每个数据集的压缩文件。

* codes/：包含我们的模型和处理函数。


## 2. 环境依赖
安装所需的包：
- pip install -r requirements.txt


## 3. 实验
运行我们的模型，请根据具体数据集执行以下命令：

cd codes/
- python main.py --dataset=fb15k-237 
- python main.py --dataset=wn18rr --max_l=5 --lr=1e-4
- python main.py --dataset=dblp --max_l=3 --gamma=0.3


## 4. 引用
    @article{nguyen2023link,
        title={Link Prediction on Latent Heterogeneous Graphs},
        author={Nguyen, Trung-Kien and Liu, Zemin and Fang, Yuan},
        journal={arXiv preprint arXiv:2302.10432},
        year={2023}
    }
## Pytorch Implementation of LFIENet

<p align="center">
  <img src="https://github.com/YT3DVision/LFIENet/blob/main/figure/network.png">
</p>
*This is a PyTorch/GPU implementation of the LFIENet.* 



### Requirements

```python
python 3.8.0
pytorch 1.13.0
cuda 12.0
h5py 
opencv-python 
pytorch_msssim
einops
```



### Installation

```
git clone https://github.com/YT3DVision/LFIENet.git
cd LFIENet
sudo pip install -r requirements.txt
```



### Quick Start

* train:

  ```
  python train.py
  ```

* validation:

  ```
  python val.py
  ```

  

### Dataset

| name  |                             Link                             |
| :---: | :----------------------------------------------------------: |
| LFIED | [gdrive](https://drive.google.com/file/d/1YiQIfqYos8FsC0azmgj3CZD7aaBiunmQ/view?usp=sharing)/[百度网盘]() |



### Citations

If LFIENet helps your research or work, please consider citing LFIENet.

```
@InProceedings{Ye_2023_TCI,
    author    = {Ye, Wuyang and Yan, Tao and Gao, Jiahui},
    title     = {LFIENet: Light Field Image Enhancement Network by Fusing Exposures of LF-DSLR Image Pairs},
    booktitle = {IEEE Transactions on Computation Imaging (TCI)},
    note	    = {DOI: 10.1109/TCI.2023.3288300},
    year      = {2023}
}
```



### Contact

If you have any questions, please contact yan.tao@outlook.com

# Python Optical Flow

This package provides python bindings to CUDA accelerated optical flow methods.

## Methods

### Brox Optical Flow
Taken from OpenCV 3.X <sup>[1](#opencv)</sup>. OpenCV must be installed on the machine.

*To do:* Write function to locate OpenCV on the machine without the use of `pkg-config`.

	@inproceedings{brox2004high,
	  title={High accuracy optical flow estimation based on a theory for warping},
	  author={Brox, Thomas and Bruhn, Andr{\'e}s and Papenberg, Nils and Weickert, Joachim},
	  booktitle={European conference on computer vision},
	  pages={25--36},
	  year={2004},
	  organization={Springer}
	}

### Fast Edge-Preserving PatchMatch for Large Displacement Optical Flow
	@inproceedings{bao2014cvpreppm,
	  title={Fast Edge-Preserving PatchMatch for Large Displacement Optical Flow},
	  author={Bao, Linchao and Yang, Qingxiong and Jin, Hailin},
	  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2014},
	  pages={3534-3541},
	  organization={IEEE}
	}

## Install
```
git clone https://github.com/linchaobao/EPPM ~/EPPM
ln -s ~/EPPM .
python2 setup.py build_ext -i
python2 demo.py
```


<a name="opencv">1</a>: https://docs.opencv.org/3.4.1/d7/d18/classcv_1_1cuda_1_1BroxOpticalFlow.html

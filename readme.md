
## FFD Source Code

Provided is code that demonstrates the training and evaluation of the work presented in the paper: "On the Detection of Digital Face Manipulation" published in CVPR 2020.

![The proposed network framework with attention mechanism](https://github.com/JStehouwer/FFD_CVPR2020/blob/master/readme_fig.png)

### Project Webpage

See the MSU CVLab website for project details and access to the DFFD dataset.

http://cvlab.cse.msu.edu/project-ffd.html

### Notes

This code is provided as example code, and may not reflect a specific combination of hyper-parameters presented in the paper.

### Description of contents

- `xception.py`: Defines the Xception network with the attention mechanism
- `train*.py`: Train the model on the train data
- `test*.py`: Evaluate the model on the test data

### Acknowledgements

If you use or refer to this source code, please cite the following paper:

	@inproceedings{cvpr2020-dang,
	  title={On the Detection of Digital Face Manipulation},
	  author={Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil Jain},
	  booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2020)},
	  address={Seattle, WA},
	  year={2020}
	}



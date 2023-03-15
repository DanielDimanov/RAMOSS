# Multi-Objective Neuroevolution for Semantic Segmentation (MOSS)

This is the official repository of [MOSS - Multi-Objective neuroevolution for Semantic Segmentation]().

In the folder dataloaders we have prepared dataloaders for some of the most popular datasets and you can start experimenting with MOSS stright away. Including:

[Cityscapes dataset](https://www.cityscapes-dataset.com/)

[MNIST dataset]()

[Fashion-MNIST dataset]()

[CIFAR10 dataset]()

[Oxford pets dataset]()

We also include `datasets.py` for even easier access to the dataloaders and example definitions to get you started even quicker.As a backbone we have used the code from [MONCAE:Multi-Objective Neuroevolution for Convolutional AutoEncoders](https://github.com/DanielDimanov/MONCAE) and [DEvol: DEvol - Deep Neural Network Evolution](https://github.com/joeddav/devol). 

For further finetuning of our CIFAR-10 models we use the repository [Training CIFAR-10 with TensorFlow2(TF2)](https://github.com/lionelmessi6410/tensorflow2-cifar), where we simply plug our mossnet discovered architecture and follow the steps provided by the authors to achieve maximum reproducability. 
 
## Setup
### Main requirements:
- tensorflow >=2.4
- numpy
- matplotlib
- pillow
- opencv-python
- pymoo
- tqdm

You can install the requirements by running `pip install -r requirements.txt`. Or you can use the `run.sh` to create a conda environment with all needed requirements. To complete the setup please use `conda activate` to activate the environement and then from within use `sh run_pip.sh` to install some of the packages not available through conda channels.

## Run the NE algorithm
To run the neuroevolution algorithm you will need to run `python run_cifar10.py` or `python run_city.py`. You are more than welcome to experiment and create more run files or help us combine everything into a single file which is well documented. 

For ease of use we have currently split it into two files so that all hyperparameters used in the original paper are conserved.

You will have to specify some of the hyperparameters when you run the scripts and in the original runs from our study we used:

`python run_city.py --max_conv_layers 25 --batch_size 4 --exp_name ${d_n}20p_20g_25l_pss10 --epochs 10 --dataset city --pop_size 20 --generations 20 --number_of_runs 1 --seed 82 --type_problem ss --pss 10
`

And

`python run_cifar10.py --dataset cifar10 --exp_name cifarExp --pop_size 10 --generations 20 --number_of_runs 1 --seed 42 --type_problem classification --batch_size 32`

## TODO:
Pull requests are welcome.
- [x] Add maximum number of conv layers as arg
- [ ] Clean up genome handlers
- [ ] Unit tests for all components

# Stability Training 

This folder contains sources that were developed to train neural network using __stability training__:
+ `network.py`: Contains a virtual `BaseNetwork` object that contains all methods to train an architecture in a supervised maner
+ `architectures.py`: Contains different architectures using `BaseNetwork` as parent.
+ `callbakcs.py`: Contains callbacks used during training for informative purpose
+ `perturbations.py`: Contains method to compute symbolic and numerical gradient w.r.t. the input
+ `utils.py`: Contains some utilities method

This folder also contains Matlab files used to transfer architectures trained on Tensorflow, on MatConvNet:
+ `load_net.m`: This method is to call to construct the network architecture.
+ `flatten_forward.m`: The forward method for a flatten operation we implemented on matconvnet (which somehow doesn't exists).
+ `flatten_backward.m`: The backward method for the flatten operation
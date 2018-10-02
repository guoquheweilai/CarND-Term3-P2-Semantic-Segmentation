# CarND-Term3-P2-Model-Semantic-Segmentation  
## Overview  
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN). You will need to extract the layers from the existing VGG-16 model and restructure the layers with several techniques like changing fully connected layer to fully convolutional layer, adding skip connections. And you will start to learn how to enhance your classifier's performance with using Intersection Over Union Metric (IOU) and inference optimization.   
Here is the link to the [orginal repository](https://github.com/udacity/CarND-Semantic-Segmentation) provided by Udaciy. This repository contains all the code needed to complete the final project for the Model Predictive Control course in Udacity's Self-Driving Car Nanodegree.  
[Placeholder]  
![um](./images/labeled_images/lr_1e-4_epoch_10_batch_size_5/um.gif)
## Prerequisites/Dependencies  
* [Python 3](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)  
* [Kitti Road dataset](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip)
    * Extract the dataset in the `data` folder. This will create the folder `data_road` with all the training and test images.
## Setup Instructions (abbreviated)  
1. Meet the `Prerequisites/Dependencies`  
2. Clone the repo from [https://github.com/udacity/CarND-Semantic-Segmentation](https://github.com/udacity/CarND-Semantic-Segmentation)  
3. Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.  
4. Build and run your code.  
5. `main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.  
## Project Description  
- [CarND-Term3-P2-Semantic-Segmentation.ipynb](./CarND-Term3-P2-Semantic-Segmentation.ipynb): Jupyter notebook for visualize coding and debugging. 
- [helper.py](./helper.py): Helper functions for use in `main.py`.  
- [main.py](./main.py): Main function to extract and restructure layers, training and validating the new classifier, then labelling the pixel of road in test images.  
- [project_tests.py](project_tests.py): Unit test functions for validating each function in `main.py`.  
- [README.md](./README.md): Writeup for this project, including setup, running instructions and project rubric addressing.  
- [images](./images): Newest inference images from `runs` folder (all images from the most recent run).  
## Run the project  
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.  
## Tips  
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
## Project Rubric  
(Placeholder)  

### 1. Build the Neural Network  
#### 1.1 Does the project load the pretrained vgg model?  
Yes, it does.  
#### 1.2 Does the project learn the correct features from the images?  
Yes, it does.  
#### 1.3 Does the project optimize the neural network?  
Yes, it does.  
#### 1.4 Does the project train the neural network?  
Yes, it does.  
### 2. Neural Network Training  
#### 2.1 Does the project train the model correctly?  
Yes, it does.  
#### 2.2 Does the project use reasonable hyperparameters?  
Yes, it does.  
#### 2.3 Does the project correctly label the road?  
Yes, it does.  
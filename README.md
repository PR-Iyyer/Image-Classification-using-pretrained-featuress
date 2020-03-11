# Image-Classification-using-pretrained-featuress
Image classification using pre-trained vectors using stacked Autoencoder and classifying using CNN.


Image Classification on CIFAR-10 dataset. 

Objective: 

To design a network combining supervised and unsupervised architectures in one model to achieve a classification task on Cifar-10 dataset under the condition that only 50% of the following classes (birds, deer, truck) can be used for training.
Solution

1. Data Exploration.

This is performed for understanding the data. 
         
There are 50000 images of shape 32 x32 x 3  flattened its pixels and stored as 50000 x 3072.  Each class has 6000 images. The images can be visualised using plotting them. 

2. Preparing training, testing and validation data. 

	Initially there are 50000 images for training and 10000 images for testing. 
           Considering the given condition, Only 50% of the images from birds, deer and trucks can be used for training. Ie, 2000 from each class in the training set will go to the test set. This will result in getting an unbalanced dataset. 

After considering this, the size of the training set will be  44000 and the testing set will be 16000. 

3. Preprocessing.

    1. Converting the data type of training and testing numpy arrays to float32 format.
    2. Normalizing
    3. One hot encoding of labels. 
    
4. Splitting 20% of training data as validation set.

5. Defining convolutional autoencoder for feature extraction.
       A stack of conv2D and its pooling layer is defined. 
       CNN takes tensors of shape(image_height, image_width,num of color channels) . Here, the model is built by two          convolutional layers stacked up with its pooling layer giving the encoder base of stacked autoencoders.
       
6. Once the model is created, it has to be compiled using loss function and an optimizer. 
       The loss function used here is mean_squared_error and optimizer is RMSprop()
       
7. Now we can visualize the model using summary() which displays all the layers and number of parameters (bias and weights).

8. Inorder to deal with the imbalance in the classes,  we use class_weight from scikit-learn library.  This will produce balanced probabilities for each class.

9. Training autoencoder
       Models can be trained using fit function. We can pass the number of epochs (Here, epoch=50), training data, batch size (here, batch size=128) and validation set. This will return a history object which can be used for evaluation.

10. Use the trained model to predict the reconstructed image using test data. 

11. Once verified, we can build a CNN for classification and use the middle layer of autoencoder as its input. 

      Hence the overall design goes like:

         
               PS: please refer to the document file in this repository..!
    


	

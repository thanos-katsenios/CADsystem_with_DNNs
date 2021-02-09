# CADsystem_with_DNNs
Binary classification of medical images with Python 3.7

This is the code for accessing a dataset that consists of images of mammograms and classifying them into benign/malignant classes. 

The main techniques that were employed are: data augmentation, feature extraction with pre-trained Convolutional Neural Networks and Support Vector Machines 
and Fine Tuning of the results.

The CROPPED_EXTRACTION file accesses the desired cropped images from the complete dataset and copies them.

The RENAME_FROM_CSV file renames the images according to their attributes from a CSV dataframe.

The CNN_CLASSIFICATION and SVM_CLASSIFICATION files display the implementation of the training and classification process.

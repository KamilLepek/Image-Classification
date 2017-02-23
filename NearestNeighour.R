# Image classifier on MNIST dataset by the nearest neighnour criterion
# About MNIST:
# http://yann.lecun.com/exdb/mnist/
# MNIST images in jpg format:
# https://drive.google.com/open?id=0B7I0L6brkUKHMlRsZXp5dUtNYlk
# This algorithm is guessing test images labels by taking the nearest train image label
# (nearest in the manhattan metric meaning)

library(jpeg)
rm(list=ls()) # clean environment
labels=9 # diffrent labels 0,1,2,3,4,5,6,7,8,9
train_n=2000 # number of each label images in training set
test_n=100  # number of each label images in test set
size = 28 # images are 28x28 pixels

mat = matrix(0,train_n*(labels+1), size*size) # each row will represent a train image

# reading train images
for(i in 0:labels)
{
  for (j in 1:train_n)
  {
    image = readJPEG(paste("/mnist_jpgfiles/train/mnist_",toString(i),"_",toString(j),".jpg",sep=""))
    temp = i*train_n+j
    mat[temp,]=image
  }
}

# mat contains now rows of train images
success=0 # number of sucessfuly guessed labels
image_numbers=(labels+1)*test_n # total amount of test images

# Guessing the labels in test images
for(i in 0:labels)
{
  for(j in 1:test_n)
  {
    image = readJPEG(paste("/mnist_jpgfiles/test/mnist_",toString(i),"_",toString(j),".jpg",sep=""))
    lf=length(image)
    mat2=matrix(rep(image,lf),train_n*(labels+1),lf,byrow=T) # each row is our test image
    diff_mat = abs(mat-mat2)
    manhattan_distance = apply(diff_mat,1,sum) # i-th element represents the manhattan distance between i-th train image and the test image
    index = which(manhattan_distance==min(manhattan_distance))#index of the image closest by manhattan metric to the test image
    label = floor((index-1)/train_n)
    if(label == i)
    {
      success = success+1
    }
  }
}

accuracy = success/image_numbers 
accuracy 
# trained on 20 000 images has about 93% accuracy on 1000 test images


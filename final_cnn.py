#!/usr/bin/env python
# coding: utf-8
'''
Name of Project:-   Object Classification using CNN
Submitted by:-      1) Sqn Ldr Vikash Kumar (Roll No: - 194102320)
                    2) Ravi Kumar Sanjay Sane (Roll No: - 194102325)
'''

# In[1]: Step 0: Import libraries

import glob, os
import numpy as np, matplotlib.pyplot as plt
from skimage import io, transform
from tensorflow.keras import layers, models, losses


# In[2]: Define other parameters and dataset paths

train_dataset_path = './training_set/'
test_dataset_path = './test_set/'
number_of_iterations = 5
number_of_classes = 2

# Resize all pics to 100*100
w = 100
h = 100
c = 3
class_names = ['dog' , 'cat']
plot_size = 4

# model save address path
model_path = './cnn_model/'


# In[3]: This function reads the dataset path and returns the dataset of pictures and labels.

def read_images(dataset_path):
    '''This function reads the dataset path and returns the dataset of pictures and labels'''
    complete_name = [dataset_path + x for x in os.listdir(dataset_path) if os.path.isdir(dataset_path + x)]
    images = []
    labels = []
    for temp_label, folder_name in enumerate(complete_name):
        for img in glob.glob(folder_name + '/*.jpg'):
            # print(f'Reading the image:{img}')            
            temp_img = io.imread(img, as_gray=False)
            temp_img = transform.resize(temp_img, (w, h, c))
            images.append(temp_img)
            labels.append(temp_label)
    return np.asarray(images, np.float32), np.asarray(labels, np.int32)


# In[4]: Step 1: Upload Dataset

''' In the given dataset training and testing are in separate folders '''
train_images, train_labels = read_images(train_dataset_path)
test_images, test_labels = read_images(test_dataset_path)

# No need to normalize the dataset as all dataset have same range
# train_images = train_images/255
# test_images = test_images/255


# In[5]: Shuffle the dataset

# to shuffle the order in which data to be fed for training
num_example = train_images.shape[0]         # the total number of pictures
arr1 = np.arange(num_example)               # np.arange (start value, end value, step size)
np.random.shuffle(arr1)                     # Rearrange the assignment after scrambling
train_images = train_images[arr1]
train_labels = train_labels[arr1]

# to shuffle the order in which data to be fed for testing
num_example = test_images.shape[0]          # the total number of pictures
arr2 = np.arange(num_example)               # np.arange (start value, end value, step size)
np.random.shuffle(arr2)                     # Rearrange the assignment after scrambling
test_images = test_images[arr2]
test_labels = test_labels[arr2]


# In[6]:

# Check that data is correctly read and show few input training data with labels
# Not so important step
plt.figure(figsize = (10, 10))
for i in range(plot_size**2):
    plt.subplot(plot_size, plot_size, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[7]: Step 2: Define the CNN architecture

# Create the convolutional base 
    
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (w, h, c)))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

# Add Dense layers on top

cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation = 'relu'))

# Final Dense layer gives output 
cnn_model.add(layers.Dense(number_of_classes))


# In[8]: Here's the complete architecture of our model.

cnn_model.summary()


# In[9]: Step 3: Compile and train the model

cnn_model.compile(optimizer = 'adam',
              loss = losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

history = cnn_model.fit(train_images, train_labels, epochs = number_of_iterations, 
                    validation_data = (test_images, test_labels))


# In[10]: Step 4: Evaluate the model

plt.plot(history.history['accuracy'], label = 'Training_accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')

test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose = 2)

print('\nTest accuracy:', test_acc)


# In[11]:
# Check and show the few of the output testing data and their predicted label

probability_model = models.Sequential([cnn_model, layers.Softmax()])
predictions = probability_model.predict(test_images)

img = test_images[:plot_size**2]
predictions_s = probability_model.predict(img)

plt.figure(figsize = (10, 10))
for i in range(plot_size**2):
    plt.subplot(plot_size, plot_size, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[np.argmax(predictions_s[i])])
plt.show()


# In[12]: Step 5: Save the model
# Save the entire model at model_path

cnn_model.save(model_path) 


# In[13]: Step 6: Load the saved model
# Reload a fresh Keras model from the given path

new_model = models.load_model(model_path)

# Check its architecture
new_model.summary()

#%%

#!/usr/bin/env python
# coding: utf-8

# ## Exercise 2
# In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.
# 
# Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.
# 
# Some notes:
# 1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
# 2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
# 3. If you add any additional variables, make sure you use the same names as the ones used in the class
# 
# I've started the code for you below -- how would you finish it? 

# In[6]:


import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
#path = f"{getcwd()}/../tmp2/mnist.npz"
path = f"{getcwd()}/mnist.npz"

# In[7]:
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    x_train = x_train/255
    x_test = x_test/255
    callback = Callback()
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE   
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation = tf.nn.relu), 
                                        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(x_train, y_train, epochs = 10, callbacks = [callback])
    # model fitting
    return history.epoch, history.history['accuracy'][-1]


# In[ ]:


train_mnist()


# In[4]:


# Now click the 'Submit Assignment' button above.
# Once that is complete, please run the following two cells to save your work and close the notebook


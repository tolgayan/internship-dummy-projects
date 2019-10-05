from keras.datasets import mnist
from keras.utils import to_categorical

from keras import models
from keras import layers
import matplotlib.pyplot as plt


''' Load keras mnist dataset and prepare it '''

# train_images: images to be used in training
# train_labels: true values of train images
# test_images : images to be used in testing
# test_labels : true values of test images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# images are 28x28. We will feed the keras model with these images. We add a last channel
# to make its shape standard as an image (28x28x1)
# divide the values with 255 (since color vals are between 0 and 255) to scale all the
# values between 0 and 1 to make training better
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype("float32") / 255

# we should convert values to a matrix.
# To understand better, check one hot encoding in image recognition
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

''' Create a keras model. Like lego pieces '''

# 28x28x1 same with the shape of our images
input_shape = (28, 28, 1)

# init a model
model = models.Sequential()

# each conv2D layer scans the whole image with a window with shape (3x3) in our case,
# and watches a feature.
# Check activation methods, and relu
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# max pooling2D scans all the image with (2x2) window and removes all cells except
# the one with the greater value.
# Without MaxPooling, our data to train would be huge and hard to train. Thanks to
# maxpooling, training operations are held better and quicker
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# By far we add layers(neurons) to detect features from the image
# Now, lets add layers to train with respect to the data comes from these
# layers:

# Flatten the images to train better
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Dropout is an operation to reduce overfitting. It is not mandatory.
# Check overfitting, and dropout
model.add(layers.Dropout(0.5))

# Since we have 10 category to predict, we enter unit value as 10,
# and by the rule, we use softmax as the activation method
model.add(layers.Dense(10, activation='softmax'))

# Lets see the general picture of our model
print(model.summary())

# Generally, rmsprop is good for any kind of model.
# loss function is again choosen by the rule
# metrics value prints current training situation to the console
model.compile(optimizer="rmsprop",
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# To start training, we call fit function.
# Check what is epoch and what is batch
history = model.fit(train_images,
                    train_labels,
                    epochs=5,
                    batch_size=32,)

# Lets print the training results as graph and you are done
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['acc']) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
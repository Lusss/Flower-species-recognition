
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
import numpy as np
# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
model_weights_path = 'vgg19_fine_tuning_16_final.h5'
# dimensions of our images.
img_width, img_height = 256, 256

test_data_dir = './test/'
nb_train_samples = 2569
nb_validation_samples = 550
epochs = 200
batch_size = 16

# build the VGG19 network
#input_tensor = Input(shape=(150, 150, 3))
input_tensor = Input(shape=(256, 256, 3))
base_model = applications.VGG19(weights='imagenet', include_top=False,input_tensor = input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024))
top_model.add(BatchNormalization())
top_model.add(Activation('relu'))
top_model.add(Dropout(0.7))

top_model.add(Dense(1024))
top_model.add(BatchNormalization())
top_model.add(Activation('relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(5, activation='softmax'))

model = Model(input=base_model.input, output=top_model(base_model.output))
model.load_weights(model_weights_path)
print('Model loaded.')

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:6]:
    layer.trainable = False
for layer in model.layers[6:]:
    layer.trainable = True
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9,decay=1e-7),
              #optimizer="adam",
              metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    color_mode="rgb",
    shuffle="false",
    class_mode=None)

filenames = test_generator.filenames
nb_samples = len(filenames)

# fine-tune the model
predict = model.predict_generator(test_generator,steps = nb_samples,verbose=1)
#y_pred = np.argmax(y_pred, axis = 1)
print(predict)
np.savetxt("prediction.csv", predict, delimiter=",")

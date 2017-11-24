
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
model_weights_path="vgg19_fine_tuning_13.h5"
# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2569
nb_validation_samples = 550
epochs = 200
batch_size = 32

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
top_model.load_weights('bottleneck_fc_model.h5')

model = Model(input=base_model.input, output=top_model(base_model.output))
print('Model loaded.')

# block the first 6 layers (up to the last conv block)
for layer in model.layers[:6]:
    layer.trainable = False
for layer in model.layers[6:]:
    layer.trainable = True
# compile the model with a SGD/momentum optimizer
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9,decay=1e-7),
              #optimizer="adam",
              metrics=['accuracy'])

# prepare data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    featurewise_center=True,
#    featurewise_std_normalization=True,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #zca_whitening=1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
tbCallBack = TensorBoard(log_dir='./GraphTest', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')

# fine-tune the model
model.fit_generator(
    train_generator,
    callbacks=[tbCallBack,checkpoint],
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

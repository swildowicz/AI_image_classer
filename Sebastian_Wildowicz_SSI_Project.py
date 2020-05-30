import os
import shutil
import zipfile

import numpy as np
from numpy.random import seed
from numpy.random import shuffle

import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
from keras.preprocessing import image


import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

DIR = '/Users/sebastian/Documents/Doktorat/Zajecia/SSI/'

class data_preparation():
  def __init__(self, dest_dir):
    self.data_zip_folder = DIR + 'data_folder.zip'
    self.data_folder = DIR + 'data_folder/'
    self.data_folder_train = DIR + 'data_folder_train'
    self.photos_turtles = self.data_folder + 'turtles/'
    self.photos_fishes = self.data_folder + 'fishes/'
    self.mixed_photos = DIR + 'mixed/'
    self.sorted_photos = DIR + 'sorted/'
    self.sorted_turtles = self.sorted_photos  + 'sorted_turtles/'
    self.sorted_fishes = self.sorted_photos  + 'sorted_fishes/'

    self.dir_list = [self.data_folder,
                      self.photos_turtles, 
                      self.photos_fishes,  
                      self.mixed_photos,
                      self.sorted_photos,
                      self.sorted_turtles,
                      self.sorted_fishes]
    self.mark_data = []
    self.sorted_data = []

  def create_folders(self):
    for dir in self.dir_list:
      if not os.path.exists(dir):
        os.mkdir(dir)

    zip_ref = zipfile.ZipFile(self.data_zip_folder , 'r')
    zip_ref.extractall(DIR)
    zip_ref.close()

    self.turtles_fnames = self.del_DS_Store(self.photos_turtles)
    self.fishes_fnames = self.del_DS_Store(self.photos_fishes)
    for i in range(len(self.turtles_fnames) + len(self.fishes_fnames)):
      self.mark_data.append(0)
      self.sorted_data.append(0)

  def clear(self):
    shutil.rmtree(self.mixed_photos)
    shutil.rmtree(self.sorted_photos)

  def mix_pictures(self):    
    seed(1)
    sequence = [i for i in range(len(self.turtles_fnames) + len(self.fishes_fnames))]
    shuffle(sequence)

    for new_fname, fname in zip(sequence[:len(self.turtles_fnames)], self.turtles_fnames):
      shutil.copyfile(self.photos_turtles + '/' + fname, self.mixed_photos + str(new_fname) + '.png')
      self.mark_data[int(new_fname)] = 1

    for new_fname, fname in zip(sequence[len(self.turtles_fnames):], self.fishes_fnames):
      shutil.copyfile(self.photos_fishes + '/' + fname, self.mixed_photos + str(new_fname) + '.png')
      self.mark_data[int(new_fname)] = 0

    self.mixed_photos_fnames = self.del_DS_Store(self.mixed_photos)

    self.mixed_images_list = []
    for fname in self.mixed_photos_fnames:
      self.mixed_images_list.append(self.mixed_photos + fname)

  def show_mixed_images(self):
    ''' Parameters for our graph; we'll output images in a 4x4 configuration '''
    nrows = 10
    ncols = 10

    ''' Set up matplotlib fig, and size it to fit 4x4 pics '''
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    for i, img_path in enumerate(self.mixed_images_list):
      ''' Set up subplot; subplot indices start at 1 '''
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)

      img = mpimg.imread(img_path)
      plt.imshow(img)
    fig.suptitle("MIXED IMAGES")
    plt.show()

  def show_ref_images(self):
    ''' Parameters for our graph; we'll output images in a 4x4 configuration '''
    nrows = 10
    ncols = 10

    ''' Set up matplotlib fig, and size it to fit 4x4 pics '''
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    images_list = []
    for fname in self.turtles_fnames:
      images_list.append(self.photos_turtles + fname)
    for fname in self.fishes_fnames:
      images_list.append(self.photos_fishes + fname)

    for i, img_path in enumerate(images_list):
      ''' Set up subplot; subplot indices start at 1 '''
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)

      img = mpimg.imread(img_path)
      plt.imshow(img)

    fig.suptitle("REFERENCES IMAGES")
    plt.show()

  def show_results(self):
    sorted_turtles = os.listdir(self.sorted_turtles)
    sorted_fishes = os.listdir(self.sorted_fishes)

    ''' Parameters for our graph; we'll output images in a 4x4 configuration '''
    nrows = 11
    ncols = 10

    ''' Set up matplotlib fig, and size it to fit 4x4 pics '''
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    images_list = []
    for fname in sorted_turtles:
      images_list.append(self.sorted_turtles + fname)

    line = [1 for m in range(10)]
    for fname in range(10):
      images_list.append(line)

    for fname in sorted_fishes:
      images_list.append(self.sorted_fishes + fname)

    for i, img_path in enumerate(images_list):
      ''' Set up subplot; subplot indices start at 1 '''
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)

      if(i + 1 > 70 and i + 1 < 81):
        plt.plot(img_path, 'r-')
      else:
        img = mpimg.imread(img_path)
        plt.imshow(img)


    fig.suptitle("SORTED IMAGES")
    plt.show()

  def del_DS_Store(self, folder):
    folder_name = folder + '/'
    list_files = os.listdir(folder)
    for fname in list_files:
      if fname == '.DS_Store':
          os.remove(folder_name + fname)

    return os.listdir(folder)
  
class AI_base():
  def __init__(self):
    pass
  
  def create_model(self):
    ''' Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G, and B '''
    img_input = layers.Input(shape=(150, 150, 3))

    ''' First convolution extracts 16 filters that are 3x3 '''
    ''' Convolution is followed by max-pooling layer with a 2x2 window '''
    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    ''' Second convolution extracts 32 filters that are 3x3 '''
    ''' Convolution is followed by max-pooling layer with a 2x2 window '''
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    ''' Third convolution extracts 64 filters that are 3x3 '''
    ''' Convolution is followed by max-pooling layer with a 2x2 window '''
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)


    ''' Flatten feature map to a 1-dim tensor so we can add fully connected layers '''
    x = layers.Flatten()(x)

    ''' Create a fully connected layer with ReLU activation and 512 hidden units '''
    x = layers.Dense(512, activation='relu')(x)

    ''' Create output layer with a single node and sigmoid activation '''
    output = layers.Dense(1, activation='sigmoid')(x)

    ''' Create model '''
    self.model = Model(img_input, output)
    ''' Print created model summary '''
    # self.model.summary()

  def train(self):
    self.model.compile(loss='binary_crossentropy',
          optimizer=RMSprop(lr=0.001),
          metrics=['acc'])

    ''' All images will be rescaled by 1./255 '''
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    ''' Flow training images in batches of 20 using train_datagen generator '''
    self.train_generator = train_datagen.flow_from_directory(
            data.data_folder_train,  # This is the source directory for training images '''
            target_size=(150, 150),  # All images will be resized to 150x150 '''
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

    ''' Flow validation images in batches of 20 using val_datagen generator '''
    self.validation_generator = val_datagen.flow_from_directory(
            data.data_folder,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

    self.history = self.model.fit(
        self.train_generator,
        steps_per_epoch = 5,  # 800 images = batch_size * steps
        epochs = 100,
        validation_data = self.validation_generator,
        validation_steps = 5,  # 100 images = batch_size * steps
        verbose = 2)

  def plot_model_history(self):
    ''' Retrieve a list of accuracy results on training and validation data sets for each training epoch '''
    acc = self.history.history['acc']
    val_acc = self.history.history['val_acc']

    ''' Retrieve a list of list results on training and validation data sets for each training epoch '''
    loss = self.history.history['loss']
    val_loss = self.history.history['val_loss']

    ''' Get number of epochs '''
    epochs = range(len(acc))

    plt.figure()
    ''' summarize history for accuracy '''
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.figure()
    ''' summarize history for loss '''
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

  def save(self):
   self.model.save('my_model.h5') 

  def load(self):
   self.model = tf.keras.models.load_model('my_model.h5')

  def process_image(self, img_path, show=False):

      img = image.load_img(img_path, target_size=(150, 150))
      img_tensor = image.img_to_array(img)                    # (height, width, channels)
      img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
      img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

      if show:
          plt.imshow(img_tensor[0])                           
          plt.axis('off')
          plt.show()
      
      return self.model.predict(img_tensor)

if __name__ == "__main__":
  print('\n\tClassification Turtles and Fishes - AI\n')
  ''' create data object '''
  data = data_preparation(DIR)
  ''' clear folders with outputs '''
  data.clear()
  ''' initialize folders if not exist '''
  data.create_folders()

  ''' print basic information about test case '''
  print('$> Reference data:')
  print('   *total turtles images:', len(data.turtles_fnames))
  print('   *total fishes images:', len(data.fishes_fnames))
  data.show_ref_images()

  ''' prepare images for AI classification - mix images '''
  print('\n\tMix images started ... ')
  data.mix_pictures()
  print('$> Total mixed images:', len(data.mixed_photos_fnames))
  data.show_mixed_images()

  print('\n$> Create AI model -  started ...')
  neuron = AI_base()

  neuron.create_model()
  neuron.train()
  # neuron.save()
  neuron.plot_model_history()

  # neuron.load()
  
  print('$> Create AI model -  success')

  ''' AI classification '''
  print('\n$> Classification started ...')
  for img, fname in zip(data.mixed_images_list, data.mixed_photos_fnames):
    prediction = neuron.process_image(img)
    if(prediction < 0.5):
      idx = fname.split('.png')
      data.sorted_data[int(idx[0])] = 0
      shutil.copyfile(img, data.sorted_fishes + '/' + fname)
    elif(prediction >=0.5):
      idx = fname.split('.png')
      data.sorted_data[int(idx[0])] = 1
      shutil.copyfile(img, data.sorted_turtles + '/' + fname)
  print('$> Classification finished\n')

  ''' check AI classification errors '''
  print('$> Total errors')
  err = 0
  total = len(data.sorted_data)

  for idx, mark, result in zip(range(len(data.mark_data)), data.mark_data, data.sorted_data):
    if mark != result:
      print ("  *Image idx: ", idx)
      err += 1

  print('$> SUM:', err)

  data.show_results()
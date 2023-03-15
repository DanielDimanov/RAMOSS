#Taken from https://keras.io/examples/vision/oxford_pets_image_segmentation/
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os

def get_img_paths(input_dir,target_dir):
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    return input_img_paths,target_img_paths

class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

# import tensorflow as tf
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt

# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# def normalize(input_image, input_mask):
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     input_mask -= 1
#     return input_image, input_mask

# @tf.function
# def load_image_train(datapoint):
#     input_image = tf.image.resize(datapoint['image'], (128, 128))
#     input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

#     if tf.random.uniform(()) > 0.5:
#         input_image = tf.image.flip_left_right(input_image)
#         input_mask = tf.image.flip_left_right(input_mask)

#     input_image, input_mask = normalize(input_image, input_mask)

#     return input_image, input_mask



# def load_image_test(datapoint):
#     input_image = tf.image.resize(datapoint['image'], (128, 128))
#     input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

#     input_image, input_mask = normalize(input_image, input_mask)

#     return input_image, input_mask

# class OxfordPetsDataset():
#     def __init__(self,batch_size,buffer_size):
#         self.TRAIN_LENGTH = info.splits['train'].num_examples
#         self.batch_size = batch_size
#         self.buffer_size = buffer_size
#         self.steps_per_epochs = self.TRAIN_LENGTH // self.batch_size
#         self.train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
#         self.test = dataset['test'].map(load_image_test)


#     def getTrainDS(self):
#         self.train_dataset = self.train.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
#         self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#         return self.train_dataset 

#     def getTestDS(self):
#         self.test_dataset = self.test.batch(self.batch_size)
#         return self.test_dataset

#     def display(self,display_list):
#         plt.figure(figsize=(15, 15))

#         title = ['Input Image', 'True Mask', 'Predicted Mask']

#         for i in range(len(display_list)):
#             plt.subplot(1, len(display_list), i+1)
#             plt.title(title[i])
#             plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#             plt.axis('off')
#         plt.show()
    




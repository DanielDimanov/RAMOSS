
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import pdb
def DatasetFromSequenceCityScapesNoHot(sequenceClass, stepsPerEpoch, nEpochs, batchSize, dims=[256,256,3], out_dims=[256,256,20], data_type=tf.float32, label_type=tf.float32):
    city_mean = (0.229, 0.224, 0.225)
    city_std = (0.485, 0.456, 0.406)
    # eager execution wrapper
    def CropAugmentEagerContext(func):
        def CropAugmentEagerContextWrapper(batchIndexTensor):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            tensors = tf.py_function(
                func,
                inp=[batchIndexTensor],
                Tout=[data_type, label_type]
            )

            # set the shape of the tensors - assuming channels last
            tensors[0].set_shape([batchSize] + dims)   # [samples, height, width, nChannels]
            tensors[1].set_shape([batchSize] + out_dims) # [samples, height, width, nClasses for one hot]
            return tensors
        return CropAugmentEagerContextWrapper
    
    @CropAugmentEagerContext
    def preprocess(batchIndexTensor):
        SEED = 42
        batchIndex = batchIndexTensor.numpy()

        # zero-based index for what batch of data to load; i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batchIndex % stepsPerEpoch
        b_images, b_masks = [],[]
        # load data
        images, masks = sequenceClass[zeroBatch]
        for image,mask in zip(images,masks):
            mask = tf.cast(mask, tf.float32)
            """
            Randomly crops image and mask in accord.
            """

#             shape = tf.cast(tf.shape(image), tf.float32)
            h = tf.cast(256, tf.int32)
            w = tf.cast(256, tf.int32)

            comb_tensor = tf.concat([image, mask], axis=2)
            comb_tensor = tf.cond(True, lambda: tf.image.random_crop(
                comb_tensor, [h, w, 3 + 20], seed=SEED), lambda: tf.identity(comb_tensor))
        #         print(comb_tensor.shape)
            image, mask = tf.split(comb_tensor, [3, 20], axis=2)
            mask = tf.cast(mask,tf.uint8)
            image = image/255
            image -= city_mean
            image /= city_std
            b_images.append(image)
            b_masks.append(mask)
            
             # convert to tensors and return
        b_images, b_masks = np.stack(b_images), np.stack(b_masks)
        b_masks = np.argmax(b_masks, axis=-1)
#         print(b_images.shape, b_masks.shape)
        return tf.convert_to_tensor(b_images), tf.convert_to_tensor(b_masks)

    
    # create our data set for how many total steps of training we have
    dataset = tf.data.Dataset.range(stepsPerEpoch*nEpochs)

    # return dataset using map to load our batches of data, use TF to specify number of parallel calls
    return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
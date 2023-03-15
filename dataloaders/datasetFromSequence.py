import tensorflow as tf
def DatasetFromSequenceClass(sequenceClass, stepsPerEpoch, nEpochs, batchSize, dims=[160,160,3], out_dims=[160,160,20], data_type=tf.float32, label_type=tf.uint8):
    # eager execution wrapper
    def DatasetFromSequenceClassEagerContext(func):
        def DatasetFromSequenceClassEagerContextWrapper(batchIndexTensor):
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
        return DatasetFromSequenceClassEagerContextWrapper

    # TF dataset wrapper that indexes our sequence class
    @DatasetFromSequenceClassEagerContext
    def LoadBatchFromSequenceClass(batchIndexTensor):
        # get our index as numpy value - we can use .numpy() because we have wrapped our function
        batchIndex = batchIndexTensor.numpy()

        # zero-based index for what batch of data to load; i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batchIndex % stepsPerEpoch

        # load data
        data, labels = sequenceClass[zeroBatch]

        # convert to tensors and return
        return tf.convert_to_tensor(data), tf.convert_to_tensor(labels)

    # create our data set for how many total steps of training we have
    dataset = tf.data.Dataset.range(stepsPerEpoch*nEpochs)

    # return dataset using map to load our batches of data, use TF to specify number of parallel calls
    return dataset.map(LoadBatchFromSequenceClass, num_parallel_calls=tf.data.experimental.AUTOTUNE)

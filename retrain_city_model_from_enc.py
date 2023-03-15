import sys
sys.path.insert(0,'..')

import numpy as np

import tensorflow as tf#
from tensorflow.keras.losses import BinaryCrossentropy
# import tensorflow_addons as tfa
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import segmentation_models as sm
from auxiliary.viz_utils import display_segmentation

input_dir = r'C:\Development\data\semantic-segmentation\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\train' #CHANGE THE PATHS!!!
target_dir = r'C:\Development\data\semantic-segmentation\cityscapes\gtFine_trainvaltest\gtFine\train' #CHANGE THE PATHS!!!
val_input_dir = r'C:\Development\data\semantic-segmentation\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\val' #CHANGE THE PATHS!!!
val_target_dir = r'C:\Development\data\semantic-segmentation\cityscapes\gtFine_trainvaltest\gtFine\val' #CHANGE THE PATHS!!!
test_input_dir = r'C:\Development\data\semantic-segmentation\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\test' #CHANGE THE PATHS!!!
test_target_dir = r'C:\Development\data\semantic-segmentation\cityscapes\gtFine_trainvaltest\gtFine\test' #CHANGE THE PATHS!!!

img_size = (1024, 2048)
num_classes = 20
batch_size = 1
import random
from dataloaders.cityscapes.cityscapes import Cityscapes,get_img_paths
from dataloaders.datasetFromSequence import DatasetFromSequenceClass 
input_img_paths,target_img_paths = get_img_paths(input_dir,target_dir)

val_input_img_paths,val_target_img_paths = get_img_paths(val_input_dir,val_target_dir)
test_input_img_paths,test_target_img_paths = get_img_paths(test_input_dir,test_target_dir)
print("Number of training samples:", len(input_img_paths))   
print("Number of validation samples:", len(val_input_img_paths))   
print("Number of testing samples:", len(test_input_img_paths))   
# Split our img paths into a training and a validation set
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths
train_target_img_paths = target_img_paths
val_input_img_paths = val_input_img_paths
val_target_img_paths = val_target_img_paths
test_input_img_paths = test_input_img_paths
test_target_img_paths = test_target_img_paths
PSS = False
if(PSS):
    from city_pss import City_PSS
    NUM_SPLITS = PSS
    city_pss_gen = City_PSS(input_dir,target_dir,img_size,num_classes,batch_size,NUM_SPLITS)
    train_gen = city_pss_gen.getSplits()
else:
    # Instantiate data Sequences for each split
    train_gen = Cityscapes(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
val_gen = Cityscapes(batch_size, img_size, val_input_img_paths, val_target_img_paths)
test_gen = Cityscapes(batch_size, img_size, test_input_img_paths, test_target_img_paths)
dataset = (train_gen, val_gen)
n_classes = 20
input_shape = (512,512,3)
batch_size = 4
TRAIN_WITH_GEN = True
TRAIN_WITH_LOGITS = True


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_sparse = tf.argmax(y_true, axis=-1)
        y_pred_sparse = tf.argmax(y_pred, axis=-1)
#         y_pred_sparse[(np.max(y_pred,axis=-1)<0.5)] = 19
        sample_weight = tf.cast(tf.less_equal(y_true_sparse, 18), tf.int32)
        return super().update_state(y_true_sparse, y_pred_sparse, sample_weight)


from datasets import get_dataset
from dataloaders.datasetFromSequence import DatasetFromSequenceClass
from dataloaders.datasetFromSequenceCityScapeNoHot import DatasetFromSequenceCityScapesNoHot
from dataloaders.datasetFromSequenceCityScapes import DatasetFromSequenceCityScapes
from dataloaders.datasetFromSequenceCityScapesFloat import DatasetFromSequenceCityScapesFloat
from dataloaders.cityscapes.city_pss import *
from auxiliary.viz_utils import *
import pickle as pkl
import math


train_gen, val_gen = dataset
# train_gen = full_train_set
# val_gen = dataset
nEpochs = 1
spatial = input_shape[0] 
training = DatasetFromSequenceCityScapesFloat(train_gen, stepsPerEpoch=len(train_gen),dims=[spatial,spatial,3], out_dims=[spatial,spatial,20], nEpochs=nEpochs, batchSize=1,).unbatch().shuffle(100).batch(batch_size)
validation = DatasetFromSequenceCityScapesFloat(val_gen,  stepsPerEpoch=len(val_gen), dims=[spatial,spatial,3], out_dims=[spatial,spatial,20], nEpochs=nEpochs, batchSize=1).unbatch().batch(batch_size)


from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


from ne import PymooGenomeReduced
TYPE_PROBLEM = 'ss'
BATCH_NORMALIZATION = True
NASWOT = False
SYNFLOW = False
GENERATIONS , POP_SIZE = 10,10
PSS = False
EPOCHS = 5
dataset, input_shape,n_classes, TRAIN_WITH_GEN, TRAIN_WITH_LOGITS, batch_size, normalize, multilabel = get_dataset('city')
input_shape = (512,512,3)
problem = PymooGenomeReduced(max_conv_layers=30, 
                                  max_dense_layers=0,
                                  max_nodes=256,
                                  max_filters=512,
                                  input_shape=input_shape,
                                  n_classes=n_classes,
                                  dropout=False,
                                  type_problem=TYPE_PROBLEM,
                                  batch_size = batch_size,
                                  TRAIN_WITH_LOGITS = TRAIN_WITH_LOGITS,
                                  batch_normalization=BATCH_NORMALIZATION,
                                  NASWOT=NASWOT,
                                  SYNFLOW=SYNFLOW
                                  )
problem.feed_data(
    train_with_gen=TRAIN_WITH_GEN,
    dataset = dataset,
    num_generations=GENERATIONS,
    pop_size = POP_SIZE,
    pss = PSS,
    multilabel=True,
    metric = 'loss',
    batch_size = batch_size,
    gen_to_tf_data = True,
    epochs=EPOCHS,
    normalize = normalize)


# Get best genomes from experiment
best_x = None
with open(r'<PATH-TO-GENERATED-POPULATION-X>', 'rb') as f:
    best_x = pkl.load(f)

# Decode genomes
models = []
for gg in best_x:
#     x = [0 if((i+3)%5==0 and i<(problem.convolution_layer_size *30)) else n for i,n in enumerate(gg)]
    x = gg
    genome = [(problem.layer_params[param][x[i_param + (i_layer*problem.convolution_layer_size)]])for i_layer in range(problem.convolution_layers) for i_param,param in enumerate(problem.convolutional_layer_shape)]
    conv_layers_len = len(genome)
    if(problem.type_problem== 'classification'):
        genome+= [(problem.layer_params[param][math.floor(x[i_param + (i_layer*problem.dense_layer_size) + conv_layers_len])])for i_layer in range(problem.dense_layers) for i_param,param in enumerate(problem.dense_layer_shape)] 
    genome += [math.floor(x[-1])]
    
    model = problem.decode(genome)[0]
    models.append(model)



LR = 0.001
optim = tf.keras.optimizers.Adam(learning_rate=LR )


dice_loss = sm.losses.DiceLoss(class_weights=[1 if i<19 else 0 for i in range(20)]) 
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), MyMeanIOU(num_classes=20)]

model.compile(optim, total_loss, metrics)
from auxiliary.lr_schs import LR_polynomial_decay
lr_sch= LR_polynomial_decay(initial_learning_rate=LR, epochs=EPOCHS, power=5)
lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_sch,verbose=1)
# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
#         tf.keras.callbacks.ModelCheckpoint('./best_so_far.h5', save_best_only=True, mode='min'),
#     tf.keras.callbacks.ReduceLROnPlateau(),
    lr_cb
]
# train model
model.fit_generator(
    training, 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=validation,
)
print('============================================================================')
print('============================================================================')
print('============================================================================')
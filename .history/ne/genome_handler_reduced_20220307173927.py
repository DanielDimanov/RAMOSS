
import gc
import math
import pickle
import operator

import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
from tqdm import tqdm

from pymoo.core.problem import ElementwiseProblem

import segmentation_models as sm

from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Softmax, GlobalAveragePooling2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras import Input,Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.python.framework import ops
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as K

from dataloaders.datasetFromSequence import DatasetFromSequenceClass 
from dataloaders.datasetFromSequenceCityScapesFloat import DatasetFromSequenceCityScapesFloat

from proxies import get_synflow_score
from losses import CITYMeanIOU

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class PymooGenomeReduced(ElementwiseProblem):
    
    def __init__(self, 
                 max_conv_layers, 
                 max_filters,
                 max_dense_layers,
                 max_nodes,
                 input_shape, 
                 n_classes,
                 batch_size=32,
                 batch_normalization=True,
                 dropout=True, 
                 max_pooling=True,
                 optimizers=None,
                 activations=None,
                 skip_ops=None,
                 type_problem='ss',
                 TRAIN_WITH_LOGITS = False,
                 NASWOT = False,
                 SYNFLOW = False,
                 smaller_ss = True,
                 min_downsample_rate = 32
                ):
        self.smaller_ss = smaller_ss
        self.min_downsample_rate = min_downsample_rate
        self.max_filters = max_filters
        self.max_nodes = max_nodes
        if max_conv_layers < 1:
            raise ValueError(
                "At least one conv layer is required for AE to work"
            )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            # 'adadelta',
            tf.keras.optimizers.Adam(learning_rate=0.001)
        ]
        self.activation = activations or [
            'relu',
            # tf.keras.activations.elu,
            tf.keras.activations.swish
        ]
        self.skip_op = skip_ops or[
            # 'none',
            'add',
            'concatenate'
        ]

        self.convolutional_layer_shape = [
            "active",
            "kernel_size",
            "activation",
            "max pooling",
            "connections",
        ]
        self.convolutional_id_to_param = {
            "active" : 0,
            "kernel_size": 1,
            "activation" : 2,
            "max pooling": 3,
            "connections": 4,
        }

        self.dense_id_to_param = {
            "active" : 0,
            "num filters" : 1,
            "activation" : 2,
        }
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2**i for i in range(int(filter_range_max-5), filter_range_max)],
            #Added after paper release
            "kernel_size": [1,3,5,7],
            "activation": list(range(len(self.activation))),
            "max pooling": list(range(2)) if max_pooling else 0,
            # Old all connections
            # "connections": [i for i in range(1,2**(max_conv_layers-1))],
            # New limitted connections
            "connections": [i for i in range(10)],
        }
        
        self.dense_layer_shape = [
            "active",
            "num filters",
            "activation",
        ]
        

        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        
        self.dense_layers = max_dense_layers
        self.dense_layer_size = len(self.dense_layer_shape)
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.type_problem = type_problem
        self.TRAIN_WITH_LOGITS = TRAIN_WITH_LOGITS
        self.i_model = 0
        
        #For archive lookup
        self.generation_performances = []
        self.generation_archive = []
        self.generation_members = []
        self.last_upsampling_index = 0

        self.NASWOT = NASWOT
        self.SYNFLOW = SYNFLOW

        self.dropout = dropout
        self.batch_norm = batch_normalization

        
        self.all_p_c = [[i for i in range(max((2**(max_conv_layers-i_layer-1))-1,0)+1)] for i_layer in range(max_conv_layers)]

        self.p_c_filtered = [
            [
            self.get_filtered_con_layer_slow(i,i_layer,self.all_p_c) for i in range(min(len(self.all_p_c[i_layer]),10))
            ] for i_layer in range(max_conv_layers)
        ]


        new_p_c_filtered =[
            [
            self.get_filtered_con_layer_fast(i,i_layer,max_conv_layers) for i in range(10)
            ] for i_layer in range(max_conv_layers)
        ]
        self.p_c_filtered = new_p_c_filtered[0:-5] + self.p_c_filtered

        import gc
        del self.all_p_c
        gc.collect()
        

        number_of_variables = (self.convolution_layers*self.convolution_layer_size) + (self.dense_layers*self.dense_layer_size) +1
        number_of_objectives = 2
        close_to_one = 0.9999999999999
        super().__init__(n_var=number_of_variables,
                         n_obj=number_of_objectives,
                         n_constr=2,
                         xl=np.array(
                             [0 for i_layer in range(self.convolution_layers) for param in self.convolutional_layer_shape] + 
                             [0 for i_layer in range(self.dense_layers) for param in self.dense_layer_shape] +
                             [0]),
                         xu=np.array(
                                    #Old connections
                                    #  [max((2**(max_conv_layers-i_layer-1))-2,0) if(param=='connections') else len(self.layer_params[param])-1 for i_layer in range(max_conv_layers) for param in self.convolutional_layer_shape] +
                                    #New connections
                                     [len(self.layer_params[param])-1 for i_layer in range(self.convolution_layers) for param in self.convolutional_layer_shape] +
                                     [len(self.layer_params[param])-1 for i_layer in range(self.dense_layers) for param in self.dense_layer_shape] +
                                     [len(self.optimizer)-1]
                                    ),
                         type_var=int)
        
    def feed_data(self,
                  train_with_gen,
                  dataset,
                  num_generations,
                  pop_size,
                  pss,
                  metric='loss',
                  batch_size = 32,
                  epochs = 5,
                  gen_to_tf_data=True,
                  multilabel=False,
                  normalize=None,
                  double_up = True
                 ):
        self.multilabel = multilabel
        self.gen_to_tf_data = gen_to_tf_data
        #Previously from NE
        self.train_tf_gen = None
        self.val_tf_gen = None
        self.train_with_gen = train_with_gen
        self.dataset = dataset
        self.num_generations = num_generations
        if(train_with_gen):
            self.train_with_gen = True
            self.train_gen,self.val_gen = dataset
        self.generation_times = []
        self.generation_performances = []
        self.generation_archive = []
        self.generation_members = []
        self.pss = pss
        self.pop_size = pop_size
        if(self.pop_size == None and self.pss != None):
            raise ValueError("Please specify pop_size if you use PSS")
        if(not self.train_with_gen):
            if len(dataset) == 2:
                (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
                self.x_val = None
                self.y_val = None
            else:
                (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = dataset
            self.x_train_full = self.x_train.copy()
            self.y_train_full = self.y_train.copy()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.normalize = normalize
        self.double_up = double_up

        
    
    def get_problem(self):
        return self.problem
    
    def _evaluate(self, x, out, *args, **kwargs):
        self.skip_model = False
         #Real version
        x_list = list(x)
        if(x_list in self.generation_members):
            #Novelty score calculation to include as an objective
            s1 = x
            novelty_score = 0
            for arch in self.generation_members:   
                sm=difflib.SequenceMatcher(None,s1,arch)
                similarity = sm.ratio()
                novelty_score+=similarity
            novelty_score = novelty_score/(len(self.generation_members))

            perf = self.generation_performances[self.generation_members.index(x_list)]
            print('Skipped evaluation')
            out['LC'] = perf['LC']
            # out["F"] = perf["F"] + [avg_similarity]
            out["F"] = perf["F"]
            # out["acc"] = perf["acc"]
            out["M"] = perf["M"]

            out["G"] = perf["G"]
        else:
            # genome = [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.convolution_layer_size)])])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.layer_params)]
            genome = [(self.layer_params[param][x[i_param + (i_layer*self.convolution_layer_size)]])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.convolutional_layer_shape)]
            conv_layers_len = len(genome)
            if(self.type_problem== 'classification'):
                genome+= [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.dense_layer_size) + conv_layers_len])])for i_layer in range(self.dense_layers) for i_param,param in enumerate(self.dense_layer_shape)] 
            genome+= [math.floor(x[-1])]

            with open(f'model_{self.i_model}.pkl','wb') as f:
                    pickle.dump(genome,f)
            
            #Decode model
            model,level_of_compression,level_of_complexity = None,None,None
            model,level_of_compression,level_of_complexity = self.decode(genome)

            v = 0
            if(self.SYNFLOW):
                raise Exception('NASWOT and SYNFLOW proxies are not part of the MOSS package and will be released at a later stage. Sorry for the inconvenience!')
            elif(self.NASWOT):
                raise Exception('NASWOT and SYNFLOW proxies are not part of the MOSS package and will be released at a later stage. Sorry for the inconvenience!')

            else:
                #Initialise performance metrics list
                performance = []
                
                #Define callbacks
                callbacks = [
                        EarlyStopping(monitor='loss', patience=0, verbose=0)
                    ]
                
                epochs = self.epochs
                
                #Define fir parameters
                if(not self.train_with_gen):
                    fit_params = {
                        'x': self.x_train_full,
                        'y': self.y_train_full,
                        'validation_split': 0.1,
                        'batch_size':self.batch_size,
                        'shuffle':True,
                        'steps_per_epoch': int(len(self.x_train_full)/self.batch_size),
                        'epochs': epochs,
                        'verbose': 1,
                        'callbacks': callbacks
                    }
                    if self.x_val is not None:
                        fit_params['validation_data'] = (self.x_val, self.y_val)
                        
                #Initialise proxy score
                sc = 0
                # print(model.summary())
                try:
                    
                    if(self.pss):
                        igen = int(self.i_model/self.pop_size)
                        if(self.gen_to_tf_data):
                            temp_tr_gen = self.train_gen[igen%self.pss]
                            nEpochs = epochs
                            out_dims = []
                            if(self.type_problem=='classification'):
                                out_dims = [self.n_classes]
                            else:
                                out_dims = list(self.input_shape)[:-1]+[self.n_classes]
                            if(self.type_problem=='ss'):
                                self.train_tf_gen = DatasetFromSequenceCityScapesFloat(temp_tr_gen, len(temp_tr_gen), nEpochs=1, batchSize=1, dims=list(self.input_shape), out_dims=out_dims).unbatch().batch(self.batch_size)
                                self.val_tf_gen = DatasetFromSequenceCityScapesFloat(self.val_gen, len(self.val_gen),  nEpochs=1, batchSize=1, dims=list(self.input_shape), out_dims=out_dims).unbatch().batch(self.batch_size)
                                performance = 0
                                if(level_of_complexity<7.7):
                                    history = model.fit(self.train_tf_gen,epochs=1, callbacks=callbacks, verbose=1)
                                    if(history.history['city_mean_iou'][-1] > 0.025):
                                        history = model.fit(self.train_tf_gen,epochs=epochs-1, callbacks=callbacks, verbose=1)
                                    if(history.history['city_mean_iou'][-1] > 0.10):
                                        performance = model.evaluate(self.val_tf_gen, steps=len(self.val_gen),verbose=1)
                                    else:
                                        performance = [10.0*history.history['loss'][-1]] + [K.epsilon()*history.history['accuracy'][-1]] + [K.epsilon()*history.history['accuracy'][-1] for _ in model.metrics]
                                else:
                                    performance = [10.0] + [K.epsilon()] + [K.epsilon() for _ in model.metrics]
                            else:
                                self.train_tf_gen = DatasetFromSequenceClass(temp_tr_gen, len(temp_tr_gen), nEpochs, self.batch_size, dims=self.input_shape, out_dims=out_dims)
                                self.val_tf_gen = DatasetFromSequenceClass(self.val_gen, len(self.val_gen), nEpochs, self.batch_size, dims=self.input_shape, out_dims=out_dims)
                                performance = 0 
                                if(level_of_complexity<7.7):
                                    history = model.fit(self.train_tf_gen,epochs=epochs, validation_data=self.val_tf_gen, validation_steps=len(self.val_gen), callbacks=callbacks, verbose=0)
                                    performance = model.evaluate(self.val_tf_gen, steps=len(self.val_gen),verbose=0)
                                else:
                                    performance = [10.0] + [K.epsilon()] + [K.epsilon() for _ in model.metrics]
                                print(performance)
                        else:
                            performance = 0 
                            if(level_of_complexity<7.7):
                                history = model.fit(self.train_gen[igen%self.pss],epochs=epochs, validation_data=self.val_gen, callbacks=callbacks, verbose=1)
                                if(history.history['accuracy'][-1] > 0.50):
                                    performance = model.evaluate(self.val_tf_gen, steps=len(self.val_gen),verbose=1)
                                else:
                                    performance = [10.0] + [K.epsilon()] + [K.epsilon() for _ in model.metrics]
                            else:
                                performance = [10.0] + [K.epsilon()] + [K.epsilon() for _ in model.metrics]
                        sc = 0
                    else:
                        if(self.train_with_gen):
                            training = self.train_gen
                            testing = self.val_gen
                            if(self.gen_to_tf_data):
                                nEpochs = self.epochs
                                out_dims = []
                                if(self.type_problem=='classification'):
                                    out_dims = [self.n_classes]
                                else:
                                    out_dims = list(self.input_shape)[:-1]+[self.n_classes]
                                if(self.type_problem=='ss'):
                                    self.train_tf_gen = DatasetFromSequenceCityScapesFloat(self.train_gen, len(temp_tr_gen), nEpochs=1, batchSize=1, dims=list(self.input_shape), out_dims=out_dims).unbatch().batch(self.batch_size)
                                    self.val_tf_gen = DatasetFromSequenceCityScapesFloat(self.val_gen, len(self.val_gen),  nEpochs=1, batchSize=1, dims=list(self.input_shape), out_dims=out_dims).unbatch().batch(self.batch_size)
                                else:
                                    self.train_tf_gen = DatasetFromSequenceClass(self.train_gen, len(self.train_gen), nEpochs, self.batch_size, dims=list(self.input_shape), out_dims=out_dims)
                                    self.val_tf_gen = DatasetFromSequenceClass(self.val_gen, len(self.val_gen), nEpochs, self.batch_size, dims=list(self.input_shape), out_dims=out_dims)
                                    self.train_tf_gen = self.train_tf_gen.shuffle(buffer_size=2000).prefetch(buffer_size=2000).cache()
                                    self.val_tf_gen = self.val_tf_gen.prefetch(buffer_size=2000).cache()
                                training =  self.train_tf_gen
                                testing =  self.val_tf_gen
                            history = None
                            performance = None
                            if(self.type_problem=='ss'):
                                history = model.fit(training,epochs=epochs,callbacks=callbacks, verbose=1)
                                performance = model.evaluate(testing,verbose=1)
                            else:
                                history = model.fit(training,epochs=epochs,callbacks=callbacks, verbose=1)
                                performance = model.evaluate(testing,verbose=1)
                            sc = 0
                        else:
                            history = model.fit(**fit_params)
                            performance = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size,verbose=1)
                            
                except Exception as e:
                    # print(e)
                    print('Broken model')
                    performance = self._handle_broken_model(model, e)
                finally:
                    if(self.gen_to_tf_data and self.train_tf_gen):
                            del self.train_tf_gen
                            del self.val_tf_gen
                            training,testing = None, None
                            del training
                            del testing
                
                v = performance[0]
                if(self.type_problem=='ss'):
                    if(performance[0]==10):
                        v = 10
                    else:
                        v = 1 - performance[-1]
            try:
                if(self.num_generations * self.pop_size > 400):
                    print('Skip saving')
                    # if(self.i_model/self.pop_size>self.num_generations-4):
                        # model.save("model-{}.h5".format(self.i_model))
                else:
                    if(level_of_complexity<7.7 and not self.SYNFLOW and performance[-1]>0.1):
                        model.save("model-{}.h5".format(self.i_model))
            except:
                print('MODEL {} COUNT NOT BE SAVED!'.format(self.i_model))
                pass
            perf = dict()
            level_of_complexity/=10
            out['LC'] = level_of_complexity
            perf['LC'] = level_of_complexity
            print(self.i_model,level_of_complexity, v)
            print('+++++++++++++++++++')
            self.i_model+=1
            out["F"] = [v,level_of_complexity]
            perf["F"] = [v,level_of_complexity]
            if(self.SYNFLOW):
                out["G"] = [15 + v, level_of_complexity-0.77]
                perf["G"] = [15 + v, level_of_complexity-0.77]
            else:
                if(self.type_problem=='ss'):
                    out["G"] = [v -0.95,  level_of_complexity-0.77]
                    perf["G"] = [v - 0.95, level_of_complexity-0.77]
                else:
                    out["G"] = [v- 1.3, level_of_complexity-0.77]
                    perf["G"] = [v - 1.3, level_of_complexity-0.77]
            # CIFAR 10 - 93.22
            # perf["G"] = [v - 1.5]
            # perf["G"] = [3 - x]
            out["M"] = "model-{}".format(self.i_model)
            perf["M"] = "model-{}".format(self.i_model)
            self.generation_members.append(x_list)
            self.generation_performances.append(perf)
            # del self.train_tf_gen
            # del self.val_tf_gen

    def get_filtered_con_layer_slow(self,i,i_layer,all_p_c):
        # Sequential 
        if(i==0):
            return int(len(all_p_c[i_layer])/2)
        else:
            # 1 and 2 = Seq + super skip (Skip to bottom 1 and 2)
            if(i<3):
                return int(len(all_p_c[i_layer])/2) + i 
    #             return -1
            # 3,4 and 5 = Skip 1-3 layers
            elif(i<6):
                return int((len(all_p_c[i_layer+(i-2)])/2))
            #Skip 2-3 layers + sequential
            elif(i<8):
                if(i_layer<len(all_p_c)-6):
                    return int(len(all_p_c[i_layer])/2) + int((len(all_p_c[i_layer+(i-4)])/2))
                else:
                    if(i==6):
                        return 3
                    else:
                        return 7
            #Inception connection - seq + skip 1 + skip 2 + skip 5
            elif(i<9):
                if(i_layer<len(all_p_c)-5):
                    return int(len(all_p_c[i_layer])/2) + int((len(all_p_c[i_layer+(i-3)])/2)) + int((len(all_p_c[i_layer+(i-6)])/2)) + int((len(all_p_c[i_layer+(i-7)])/2))
                else:
                    # Can't inception, so next best is dense connect
                    return max(all_p_c[i_layer])
            #dense block of next 5
            else:
                if(i_layer<len(all_p_c)-5):
                    return int(len(all_p_c[i_layer])/2) + int((len(all_p_c[i_layer+(i-8)])/2)) + int((len(all_p_c[i_layer+(i-7)])/2)) + int((len(all_p_c[i_layer+(i-6)])/2)) + int((len(all_p_c[i_layer+(i-5)])/2))
                else:
                    # Can't dense 5, so next best is all dense - the last because of i8 repetition
                    return all_p_c[i_layer][-2:][0]

    def get_filtered_con_layer_fast(self,i,i_layer,max_conv_layers):
            base = 2**(max_conv_layers-i_layer-1)
            # Sequential 
            if(i==0):
                return int(base/2)
            else:
                # 1 and 2 = Seq + super skip (Skip to bottom 1 and 2)
                if(i<3):
                    return int(base/2) + i 
        #             return -1
                # 3,4 and 5 = Skip 1-3 layers
                elif(i<6):
                    temp = 2**(max_conv_layers-(i_layer+(i-2))-1)
                    return int(temp/2)
                #Skip 2-3 layers + sequential
                elif(i<8):
                    if(i_layer<max_conv_layers-6):
                        temp = 2**(max_conv_layers-(i_layer+(i-4))-1)
                        return int(base/2) + int(temp/2)
                    else:
                        if(i==6):
                            return 3
                        else:
                            return 7
                #Inception connection - seq + skip 1 + skip 2 + skip 5
                elif(i<9):
                    if(i_layer<max_conv_layers-5):
                        temp_1 = 2**(max_conv_layers-(i_layer+(i-3))-1)
                        temp_2 = 2**(max_conv_layers-(i_layer+(i-6))-1)
                        temp_3 = 2**(max_conv_layers-(i_layer+(i-7))-1)
                        return int(base/2) + int(temp_1/2) + int(temp_2/2) + int(temp_3/2)
                    else:
                        # Can't inception, so next best is dense connect
                        return base
                #dense block of next 5
                else:
                    if(i_layer<max_conv_layers-5):
                        temp_1 = 2**(max_conv_layers-(i_layer+(i-8))-1)
                        temp_2 = 2**(max_conv_layers-(i_layer+(i-7))-1)
                        temp_3 = 2**(max_conv_layers-(i_layer+(i-6))-1)
                        temp_4 = 2**(max_conv_layers-(i_layer+(i-5))-1)
                        return int(base/2) + int((temp_1/2)) + int((temp_2/2)) + int((temp_3/2)) + int((temp_4/2))
                    else:
                        # Can't dense 5, so next best is all dense - the last because of i8 repetition
#                         return all_p_c[i_layer][-2:][0]
                        return base-2
    

    def decode(self, genome):
        self.down_ids = []
        self.up_ids = []
        # print([genome[8 + i*self.convolution_layer_size] for i in range(self.convolution_layers) if genome[i*convolution_layer_size]==1])
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")

        active_layers = len([0 for i in range(self.convolution_layers) if genome[i*self.convolution_layer_size]==1])
        p_c_individual = self.p_c_filtered[-active_layers:]
        cons = [genome[self.convolutional_id_to_param['connections'] + i*self.convolution_layer_size] for i in range(self.convolution_layers) if genome[i*self.convolution_layer_size]==1]
        cons = [p_c_individual[i_l][con] if(len(p_c_individual[i_l])>con) else p_c_individual[i_l][0] for i_l,con in enumerate(cons)]
        lays = []
        x = None
        dim = 0
        offset = 0
        optim_offset = 0
        if self.convolution_layers > 0:
            dim = min(self.input_shape[:-1])  # keep track of smallest dimension
        input_layer = True
        dims = []
        gateways = dict()
        temp_features = self.max_filters
        min_features = 16
        if(self.double_up):
            # temp_features = int(self.max_filters/self.convolution_layers)
            # temp_features = 32
            temp_features = min_features
        features = dict()
        naswot_outputs = []
        encoder_max_features = temp_features

        x = Input(shape=self.input_shape,dtype=tf.float32)
        add_layer(cons,lays, x, 0)
        for i in range(self.convolution_layers):
            if genome[offset]:
                if input_layer:
                    if(not self.smaller_ss):
                        temp_features = genome[offset + self.convolutional_id_to_param['num filters']]
                    temp_kernel = genome[offset + self.convolutional_id_to_param['kernel_size']]
                    x =  Convolution2D(
                        temp_features, (temp_kernel, temp_kernel),
                        padding="same",
                        # V 7 new
                        # kernel_initializer = 'he_normal',
                        # kernel_initializer = tf.keras.initializers.GlorotNormal(),
                        # bias_initializer = tf.keras.initializers.Constant(0.1),
                        # V12
#                         input_shape=self.input_shape,
                        name=f'conv2d_l{i}_n',
                        # activation=self.activation[genome[offset + self.convolutional_id_to_param['activation']]],
                        # V15 added regularisation
                        # kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005)
                    )
                    lays.append(x)
                    input_layer = False
                else:
                    if(not self.smaller_ss):
                        if(self.type_problem =='classification'):
                            temp_features = genome[offset + self.convolutional_id_to_param['num filters']]
                        else:
                            temp_features = int(min(features[list(features.keys())[-1]],genome[offset + self.convolutional_id_to_param['num filters']]))
                    temp_kernel = genome[offset + self.convolutional_id_to_param['kernel_size']]
                    x = Convolution2D(
                        temp_features, (temp_kernel, temp_kernel),
                        padding="same",
                        name=f'conv2d_l{i}_encoder',
                        # kernel_initializer = 'he_normal',
                        # activation=self.activation[genome[offset + self.convolutional_id_to_param['activation']]],
                        #V15 added regularisation
                        # kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005)
                    )
                    lays.append(x)
                # if genome[offset + self.convolutional_id_to_param['batch normalisation']]:
                #     x = BatchNormalization()
                #     add_layer(cons,lays,x,len(lays))
                if(self.batch_norm):
                    x = BatchNormalization()
                    add_layer(cons,lays,x,len(lays))
                x = Activation(self.activation[genome[offset + self.convolutional_id_to_param['activation']]])
                add_layer(cons,lays,x,len(lays))
                #Append the gateway to layer for skip connection BEFORE pooling
                # if(not self.skip_op[genome[offset+7]]=='none'):
                #     gateways[offset]=((dim,x))
                max_pooling_type = genome[offset + self.convolutional_id_to_param['max pooling']]
                if max_pooling_type == 1 and self.input_shape[1]/dim < self.min_downsample_rate:
                    temp_kernel = genome[offset + self.convolutional_id_to_param['kernel_size']]
                    if(self.double_up and temp_features<self.max_filters):
                        temp_features *= 2
                    x = MaxPooling2D(pool_size=(2, 2), padding="same")
                    # x = Convolution2D(temp_features, (temp_kernel,temp_kernel), strides=2,  #V15 added regularisation 
                        # kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005),
                        # padding="same")
                    add_layer(cons,lays,x,len(lays))
                    dim /= 2
                # Added dropout 2021/12/20
                if(self.dropout):
                    x = Dropout(0.2)
                    add_layer(cons,lays,x,len(lays))
            dims.append(dim)
            features[i] = temp_features
            dim = int(math.ceil(dim))
            if(i<self.convolution_layers-1):
                offset += self.convolution_layer_size
            else:
                optim_offset = offset + self.convolution_layer_size
                encoder_max_features = temp_features
        # x = Convolution2D(temp_features,(3,3),padding="same", kernel_initializer='he_uniform')
        # add_layer(cons,lays,x,len(lays))
        # level_of_compression = np.prod(x.get_shape()[1:])
        #level_of_compression is limited to 10 instead of 5 in the original MONCAE paper!
        # level_of_compression = min(math.log(level_of_compression,10),10)
        # TODO TEMP disbaled loc
        level_of_compression = 10
        needed_reductions = [i-2 for i,temp_dim in enumerate(dims) if(math.ceil(temp_dim)!=math.floor(temp_dim))]
        if(self.type_problem=='classification'):
            # x = GlobalAveragePooling2D()
            x = Flatten()
            add_layer(cons,lays,x,len(lays))
        if(self.type_problem=='ss' or self.type_problem=='ae'):
            stop_decoder = 3
            #Reset the offset
            current_features = encoder_max_features
            for i in reversed(range(self.convolution_layers)):
                #Done to fix shape when 14->7-> 4 => 4->8->16->14
                if(not(dim in dims) and ((dim-2)*2 in dims or (not(dim*2 in dims) and (dim-2)*2==min(self.input_shape[:-1])))):
                    x = Convolution2D(current_features,(3,3))
                    add_layer(cons,lays,x,len(lays))
                    dim-=2
                if(stop_decoder==0):
                    continue
                elif genome[offset]:
                    skipped = False
                    temp_kernel = genome[offset + self.convolutional_id_to_param['kernel_size']]
                    max_pooling_type = genome[offset + self.convolutional_id_to_param['max pooling']]
                    if(dim ==self.input_shape[1]):
                        current_features = self.n_classes * 2
                        stop_decoder -= 1
                    x = Convolution2D(
                        current_features, (temp_kernel, temp_kernel),
                        padding="same",
                        name=f'conv2d_l{i}_decoder',
                        # kernel_initializer = 'he_normal',
                        # # V7
                        # kernel_initializer = tf.keras.initializers.GlorotNormal(),
                        # bias_initializer = tf.keras.initializers.Constant(0.1),
                        # DISABLED ON 21.02.2022
                        # activation=self.activation[genome[offset + self.convolutional_id_to_param['activation']]],
                    )
                    add_layer(cons,lays,x,len(lays))
                    if(self.batch_norm):
                        x = BatchNormalization()
                        add_layer(cons,lays,x,len(lays))
                    x = Activation(self.activation[genome[offset + self.convolutional_id_to_param['activation']]])
                    add_layer(cons,lays,x,len(lays))
                    # if(not self.type_problem=='ss'):
                    #     x = Activation(self.activation[genome[offset + 4]])
                    #     add_layer(cons,lays,x,len(lays))
                    if (((dim*2 in dims or (dim*2)-2 in dims or (dim*4)-2 or dim==(int(min(self.input_shape[:-1]))/2))) and dim<min(self.input_shape[:-1])):
                        x = UpSampling2D((2, 2))
                        # if(current_features>min_features*2):
                        current_features=math.ceil(current_features/2)
                        # x = Conv2DTranspose(current_features, (temp_kernel,temp_kernel), strides= 2, padding="same")
                        add_layer(cons,lays,x,len(lays))
                        self.last_upsampling_index = len(lays)-1
                        dim*=2
                    # if(self.batch_norm):
                    #     x = BatchNormalization()
                    #     add_layer(cons,lays,x,len(lays))
                if(dim>max(self.input_shape)):
                    import pdb
                    pdb.set_trace()
                offset -= self.convolution_layer_size

            if(not self.type_problem=='ss'):
                x = Convolution2D(self.input_shape[-1], (genome[self.convolutional_id_to_param['kernel_size']],genome[self.convolutional_id_to_param['kernel_size']]), activation=self.activation[genome[self.convolutional_id_to_param['activation']]], padding="same", )
                add_layer(cons,lays,x,len(lays))
            else:
                if(self.SYNFLOW):
                    x = Convolution2D(self.n_classes, self.input_shape[-1], activation=None, padding="same",)
                else:
                    x = Convolution2D(self.n_classes, self.input_shape[-1], activation="softmax", padding="same",)
                add_layer(cons,lays,x,len(lays))
            
        #Clear connections
        dirty_cons = None
        clean_cons = None
        # print(lays)
        try:
            # import pdb
            # pdb.set_trace()
            if(self.type_problem=='ss'):
                cons = add_unet_cons(cons,lays)
            
            dirty_cons = decode_connections(cons,len(cons))
            clean_cons = clear_cons(dirty_cons,len(cons))
        except Exception as ex:
            print(cons,len(cons))
            print('Failed connections!')
            print(ex)
            
        operations = []
        operations = self.decode_ops(operations,lays,clean_cons)
        if(self.type_problem=='classification'):
            ### Dense layer decoding (classification only)
            
            offset = optim_offset
            has_a_dense_active = False
            for i in range(self.dense_layers):
                if genome[offset]:
                    temp_nodes = genome[offset + self.dense_id_to_param['num filters']]
                    x =  Dense(
                        temp_nodes,
                        # kernel_initializer = 'he_normal',
                        # # V7
                        # kernel_initializer = tf.keras.initializers.GlorotNormal(),
                        # bias_initializer = tf.keras.initializers.Constant(0.1),
                        # activation=self.activation[genome[offset + self.dense_id_to_param['activation']]]
                    )
                    operations.append(x(operations[-1]))
                    dr =  Dropout(0.2)
                    operations.append(dr(operations[-1]))


                    has_a_dense_active = True
                offset+= self.dense_layer_size
            if(not has_a_dense_active):
                x = Dense(self.n_classes)
                operations.append(x(operations[-1]))
            if(self.SYNFLOW):
                x = Dense(self.n_classes)
            elif(self.multilabel == True):
                x = Dense(self.n_classes, activation='sigmoid')
            else:
                x = Dense(self.n_classes, activation='softmax')
            operations.append(x(operations[-1]))
            optim_offset = offset
        outs = list()  
        # import pdb
        # pdb.set_trace()
        if(self.NASWOT):
            outs = [op for op in operations if 'elu' in op.name or 'wish' in op.name]
            if(len(outs)<1):
                outs=operations[-1]
        else:
            outs=operations[-1]
        tf.keras.backend.clear_session()
        model = Model(operations[0],outs)
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='af.png', show_shapes=True)
        # TODO changed from binary_crossentropy
        metrics = ["accuracy"]
        # print(self.type_problem, self.TRAIN_WITH_LOGITS)
        if(self.TRAIN_WITH_LOGITS):
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        elif(self.type_problem=='ss' or self.type_problem=='ae'):
            loss = 'sparse_categorical_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        if(self.type_problem =='ss'):
            dice_loss = sm.losses.DiceLoss(class_weights=[1 if i<19 else 0 for i in range(20)]) 
            focal_loss = sm.losses.BinaryFocalLoss() if self.n_classes == 1 else sm.losses.CategoricalFocalLoss()
            loss = dice_loss + (1 * focal_loss)
            metrics += [CITYMeanIOU(num_classes=self.n_classes)]
        #TODO CHANGE THAT!!!!
        if(True):
            id_to_name={
                0:'Gun',
                1:'Knife',
                2:'Wrench',
                3:'Pliers',
                4:'Scissors'
            }
            # metrics+= [tf.keras.metrics.Precision(class_id=idx, name='precision_{}'.format(id_to_name[idx])) for idx in range(len(id_to_name))]
        if(not self.NASWOT and not self.SYNFLOW):
            model.compile(loss=loss,
                        optimizer=self.optimizer[genome[optim_offset]],
                        metrics=metrics)
        # num_params = np.sum([np.prod(l.output_shape[1:],dtype=np.int64) for l in model.layers], dtype=np.int64)
        num_params = model.count_params()
        # level_of_complexity = min(math.log(int(model.count_params()),10),10)
        level_of_complexity = min(math.log(int(num_params),10),10)
        return model,level_of_compression,level_of_complexity
    
    def decode_ops(self,operations=list(),lays=list(),cons=list()):
        index_fixer = len(operations)
        operations = operations.copy()
        for index,con in enumerate(cons.transpose()):
            #First layer in cell
            if(index==0):
                operations.append(lays[0])
            else:
                #TODO FIND MORE OPTIMIAL WAY OF DOING THIS!!!
                nz = np.nonzero(con)[0]
                if(len(nz)>1):
                    shapes = {operations[layer+index_fixer].shape[-1] for layer in nz}
                    k_shapes = {operations[layer+index_fixer].shape[1] for layer in nz}
                    full_shapes = {(operations[layer+index_fixer].shape[1],operations[layer+index_fixer].shape[3]) for layer in nz}
                    can_add = True
                    should_conc = False
                    if(len(full_shapes)==1):
                        op = add([operations[layer+index_fixer] for layer in nz])
                    else:
                        # print('Gonna try')
                        adjustment_ops = []
                        layers_for_addition = [operations[layer+index_fixer] for layer in nz]
                        lowest_dim_ind = 0
                        lowest_dim = layers_for_addition[0].shape[1]
                        lowest_number_of_features = layers_for_addition[-1].shape[-1]
                        if(self.type_problem=='ss'):
                            lowest_dim = layers_for_addition[len(layers_for_addition)-1].shape[1]
                            lowest_dim_ind = len(layers_for_addition)-1
                        else:
                            for i_l,l in enumerate(layers_for_addition):
                                if(l.shape[1]<lowest_dim):
                                    lowest_dim = l.shape[1]
                                    lowest_dim_ind = i_l
                        for i_l,l_to_add in enumerate(layers_for_addition):
                            adjust_op = l_to_add
                            tries = 0
                            adjustment_ops_len = len(adjustment_ops)
                            # Temp disable to attempt Reshape
                            while(adjust_op.shape[1] != layers_for_addition[lowest_dim_ind].shape[1] and can_add):
                                if(adjust_op.shape[1]<layers_for_addition[lowest_dim_ind].shape[1]):
                                    adjust_op = UpSampling2D((2, 2), name=f'adj_{index}_{adjustment_ops_len}_{tries}_{i_l}_up')(adjust_op)
                                else:
                                    adjust_op = Convolution2D(lowest_number_of_features,kernel_size=(3,3), strides=2, padding="same", name=f'adj_{index}_{adjustment_ops_len}_{tries}_{i_l}_down')(adjust_op)
                                tries+=1
                                if(tries>10):
                                    can_add=False
                                    print('Cannot add {} and {} and {}'.format(l_to_add.shape, adjust_op.shape ,layers_for_addition[lowest_dim_ind].shape))
                            if(i_l!=lowest_dim_ind):
                                should_conc = True
                                adjust_op = Convolution2D(layers_for_addition[lowest_dim_ind].shape[-1],kernel_size=(1, 1), padding="same")(adjust_op)
                            # adjust_op = Reshape(( layers_for_addition[lowest_dim_ind].shape[1], layers_for_addition[lowest_dim_ind].shape[2], int(adjust_op.shape[3]*((adjust_op.shape[1]/layers_for_addition[lowest_dim_ind].shape[1])**2))))(adjust_op)
                            # adjust_op = Convolution2D(1, kernel_size=(1,1), padding="same")(adjust_op)
                            # adjust_op = Dropout(0.5) (adjust_op)
                            adjustment_ops.append(adjust_op)
                        if(can_add):
                            # Concatenation disabled temporary!!!
                            if(should_conc):
                                op = concatenate([adj_op for adj_op in adjustment_ops])
                            else:
                                op = add([adj_op for adj_op in adjustment_ops])
                            # op = concatenate([adj_op for adj_op in adjustment_ops])
                        # print('We did it?')
                    # elif((len(k_shapes)==1) and len(shapes)!=1):
                    #     desired_shape = operations[-1].shape[-1]
                    #     op = concatenate([operations[layer+index_fixer] for layer in nz])
                    #     #Fix shape with identity
                    #     op = Convolution2D(desired_shape,(1,1),padding="same")(op)
                    if(can_add == False):
                        operations.append(lays[index](operations[-1]))
                        continue
                    else:
                        operations.append(lays[index](op))
                elif(len(nz)==1):
                    operations.append(lays[index](operations[-1]))
                else:
                    print('======ERRORRORORORR ========')
#                     pdb.set_trace()
                    continue
        return operations
    
    def decode_model_genome(self, genome):
        x = genome
        genome = [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.convolution_layer_size)])])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.layer_params)]
        if(self.type_problem== 'classification'):
            genome+= [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.dense_layer_size)])])for i_layer in range(self.dense_layers) for i_param,param in enumerate(self.dense_layer_shape)]
            
        genome+= [math.floor(x[-1])]
        
        #Decode model
        try:
            model,level_of_compression,level_of_complexity = self.decode(genome)
        except:
            raise Exception('Model could not be decoded')
        return model
        
    
    def convParam(self, i):
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]
    
    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]
        
    def is_compatible_genome(self, genome):
        expected_len = (self.convolution_layers * self.convolution_layer_size) + (self.dense_layers * self.dense_layer_size) + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convParam(j):
                    return False
            ind += self.convolution_layer_size
            
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
            
        if genome[ind] not in range(len(self.optimizer)):
            return False
        
        return True
    
    def _handle_broken_model(self, model, error):
        self.skip_model = True
        print('================')
        print('Number of parameters:', str(model.count_params()))
        print('================')

        n = self.n_classes
        # v2 Added loss 10 times more for models out of score to make them infavourable
        # performance = [log_loss(np.concatenate(([1], np.zeros(n - 1))), np.ones(n) / n)*10, math.log((self.input_shape[1]*self.input_shape[1]),10), 1]
        performance = [10.0] + [K.epsilon()] * (len(model.metrics)-1)
        # del model
        gc.collect()

        if K.backend() == 'tensorflow':
            K.clear_session()
            #Changed from tensorflow
            ops.reset_default_graph()

        print('An error occurred and the model could not train!')
        print(('Model assigned poor score. Please ensure that your model'
               'constraints live within your computational resources.'))
        return performance
    
    

        
# Fixed add layer 16.02.2022 
def add_layer(cons, lays, layer,pos):
    # print('> cons before', cons)
    size = len(cons)
    con = int(2**((size)-(pos+1)))
    cons.insert(pos,con)
    lays.insert(pos,layer)
    if(cons[-2]==0):
        cons[-2] = 1
    for i in range(0,pos):
        cons[i] = [digit for digit in bin(cons[i])[2:]]
        cons[i].insert(pos,'0')
        cons[i] = int(''.join(cons[i]),2)
    return cons, lays

def decode_connections(cons,cell_size):
    bin_cons = list()
    for i_con,con in enumerate(cons):
        overflow = 2**(cell_size)
        tries = 0
        enough = False
        while(cons[i_con]>=overflow and not enough):
            cons[i_con] -= overflow
            tries+=1
            if(tries>cell_size*10):
                enough = True

    for con in cons:
        binarised = bin(con)[2:]
        bin_cons.append([int(digit) for digit in eval("f\"{binarised:0>"+str(cell_size)+"}\"")])
    # print(bin_cons)
#     bin_cons.append([0] * cell_size)
    return np.array(bin_cons, dtype=object)

def clear_cons(dirty_cons,cell_size):
    # Disabled v4

    # b = np.ones(cell_size)
    # np.fill_diagonal(dirty_cons[:,1:], b)
    clean_cons = np.triu(dirty_cons, k=1)
    for i in range(1,len(clean_cons)):
        # Added v5 to prevent disconnected layer which was connected but in a wrong way
        if(1 not in clean_cons[i,:] and 1 in dirty_cons[i,:]):
            clean_cons[i,i+1] = 1
        if(1 not in clean_cons[:,i]):
            # Changed on 16/02/2022 from:
            # clean_cons[0,i] = 1 SUPER FAULTY
            # To:
            clean_cons[i-1,i] = 1
    return clean_cons


def add_unet_cons(cons,lays):
    down_ids = [i-1 for i,l in enumerate(lays) if(((hasattr(l, 'strides') and l.strides[0]==2) or 'pooling' in l.name)  and not 'ranspose' in l.name and not 'adj' in l.name)][1:]
    up_ids = [i+2 for i,l in enumerate(lays) if(('ranspose' in l.name or 'up' in l.name) and not 'adj' in l.name)][::-1][1:]
    for down_id, up_id in zip(down_ids,up_ids):
        #Check if already is there and only add if it is not!
        already_connected = [i_d for i_d,d in enumerate(bin(cons[down_id])) if d=='1']
        if(not up_id in already_connected):
            cons[down_id] += 2**(len(lays)-up_id)
    return cons
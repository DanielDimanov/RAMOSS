import numpy as np
# from pymoo.util.misc import stack
from pymoo.core.problem import ElementwiseProblem

    
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
import random as rand
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add,concatenate,Dot
from tensorflow.keras import Input,Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.framework import ops
from sklearn.metrics import log_loss
import gc
import tensorflow as tf
from losses import dice_loss
from nasbench import api
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

class PymooNASBenchGenome(ElementwiseProblem):
    
    def __init__(self, 
                 max_conv_layers, 
                 max_filters,
                 input_shape, n_classes,
                 batch_size=256,
                 batch_normalization=True,
                 dropout=True, 
                 max_pooling=True,
                 optimizers=None,
                 activations=None,
                 skip_ops=None,
                 type_problem='classification',
                 TRAIN_WITH_LOGITS = False,
                 nasbench_api = None
                ):
        if nasbench_api is None:
            import pdb
            pdb.set_trace()
            # raise ValueError(
            #     "You need to pass the nasbench api"
            # )
        if max_conv_layers < 1:
            raise ValueError(
                "At least one conv layer is required for AE to work"
            )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0

        self.convolutional_layer_shape = [
            "active",
            "kernel_size",
            "max pooling",
            "connections",
            # "skip_op"
        ]
        self.convolutional_id_to_param = {
            "active" : 0,
            "kernel_size": 1,
            "max pooling": 2,
            "connections": 3,
            # "skip_op": 5
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
            "kernel_size": [1,3],
            "max pooling": list(range(2)) if max_pooling else 0,
            # Old all connections
            "connections": [i for i in range(1,2**(max_conv_layers-1))]
        }

        self.nasbench_api = nasbench_api
        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        
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

        # self.NASWOT = NASWOT
        # self.SYNFLOW = SYNFLOW

        self.dropout = dropout
        self.batch_norm = batch_normalization



        # conv_layers*conv_layer_size + 1 variables
        # 3 objectives = loss, compression, complexity
        number_of_variables = (self.convolution_layers*self.convolution_layer_size)
        number_of_objectives = 2
        close_to_one = 0.9999999999999
        super().__init__(n_var=number_of_variables,
                         n_obj=number_of_objectives,
                         n_constr=0,
                         xl=np.array(
                             [0 for i_layer in range(self.convolution_layers) for param in self.convolutional_layer_shape]),
                         xu=np.array(
                                    #Old connections
                                    #  [max((2**(max_conv_layers-i_layer-1))-2,0) if(param=='connections') else len(self.layer_params[param])-1 for i_layer in range(max_conv_layers) for param in self.convolutional_layer_shape] +
                                    #New connections
                                     [len(self.layer_params[param])-1 for i_layer in range(max_conv_layers) for param in self.convolutional_layer_shape]
                                    ),

                         type_var=int)
        
        
    
    def get_problem(self):
        return self.problem
    
    def _evaluate(self, x, out, *args, **kwargs):
         #Real version
        #  genome = [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.convolution_layer_size)])])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.layer_params)]
        genome = [(self.layer_params[param][x[i_param + (i_layer*self.convolution_layer_size)]])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.convolutional_layer_shape)]
        conv_layers_len = len(genome)

        model_nas,level_of_compression,level_of_complexity = self.decode_nasbench(genome)
        
        #Decode model
        query = self.nasbench_api.query(model_nas)
        level_of_complexity = min(math.log(int(query['trainable_parameters']),10),10)
            
        #Initialise performance metrics list
        training_time = query['training_time']
        val_acc = query['validation_accuracy']
        test_acc = query['test_accuracy']
        self.i_model+=1
        out["acc"] = [val_acc,test_acc] 
        out["TT"] = [training_time]
        out["Q"] = query
        out["M"] = "model-{}".format(self.i_model)
#         out["F"] = [1-val_acc, level_of_complexity/10]
        out["F"] = [1-val_acc,level_of_complexity/10]
    
    
    def decode_nasbench(self, genome):
        active_layers = len([0 for i in range(self.convolution_layers) if genome[i*self.convolution_layer_size]==1])
        cons = [genome[self.convolutional_id_to_param['connections'] + i*self.convolution_layer_size] for i in range(self.convolution_layers) if genome[i*self.convolution_layer_size]==1]
        num_pools = 0
        num_active_convs = len(cons)
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
        temp_features = 0
        features = dict()

        x = INPUT
        add_layer(cons,lays, x, 0)
        for i in range(self.convolution_layers):
            if genome[offset]:
                
                x = CONV1X1 if(genome[offset + self.convolutional_id_to_param['kernel_size']]==1) else CONV3X3
                lays.append(x)
                input_layer = False
                #Append the gateway to layer for skip connection BEFORE pooling
                max_pooling_type = genome[offset + self.convolutional_id_to_param['max pooling']]
                #Len lays<5 because of NASBENCH
                if max_pooling_type == 1 and num_pools<5-num_active_convs:
                    x = MAXPOOL3X3
                    add_layer(cons,lays,x,len(lays))
                    dim /= 2
                    num_pools += 1
            dims.append(dim)
            features[i] = temp_features
            dim = int(math.ceil(dim))
            if(i<self.convolution_layers-1):
                offset += self.convolution_layer_size
            else:
                optim_offset = offset + self.convolution_layer_size
        #Reset the offset
        
        x = OUTPUT
        add_layer(cons,lays,x,len(lays))
        #Clear connections
        dirty_cons = None
        clean_cons = None
        try:
            dirty_cons = decode_connections(cons,len(cons))
            clean_cons = clear_cons(dirty_cons,len(cons))
        except:
            print('Failed cons!')
            import pdb
            pdb.set_trace()
        #NAS LIMITATION 9 connections
        if(np.unique(clean_cons,return_counts=True)[1][1]>9):
            cons_np = np.array(clean_cons)
            ctocs = np.argwhere(cons_np.transpose())
            nc = len(ctocs)
            n_r = 0
            while nc>9:
                n_r = n_r%len(cons_np)
                idxs = np.where(ctocs[:,0]==n_r)[0]
                if(len(idxs)>1):
                    idx = 0
                    cons_np[ctocs[idxs[idx]][1]][ctocs[idxs[idx]][0]] = 0
                    ctocs = np.delete(ctocs,idxs[idx],axis=0)
                    nc = len(ctocs)
                n_r += 1
                
            clean_cons = cons_np.tolist()
            print('reduced cons')
        level_of_compression = 1
        # Claculated at evaluate
        level_of_complexity = 1
        model_nas = api.ModelSpec( matrix= clean_cons, ops=lays)
        return model_nas,level_of_compression,level_of_complexity
    
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
    
    def get_genome_int(self,genome_float):
        genome = [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.convolution_layer_size)])])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.layer_params)]
        return genome
    
    def decode_model_genome(self, genome):
        x = genome
        genome = [(self.layer_params[param][math.floor(x[i_param + (i_layer*self.convolution_layer_size)])])for i_layer in range(self.convolution_layers) for i_param,param in enumerate(self.layer_params)]
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
        
    def is_compatible_genome(self, genome):
        expected_len = self.convolution_layers * self.convolution_layer_size 
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convParam(j):
                    return False
            ind += self.convolution_layer_size
        return True
    
    def _handle_broken_model(self, model, error):
        print('================')
        print('Number of parameters:', str(model.count_params()))
        print('================')
        del model

        n = self.n_classes
        # v2 Added loss 10 times more for models out of score to make them infavourable
        performance = [log_loss(np.concatenate(([1], np.zeros(n - 1))), np.ones(n) / n)*10, math.log((self.input_shape[1]*self.input_shape[1]),10)]
        gc.collect()

        if K.backend() == 'tensorflow':
            K.clear_session()
            #Changed from tensorflow
            ops.reset_default_graph()

        print('An error occurred and the model could not train!')
        print(('Model assigned poor score. Please ensure that your model'
               'constraints live within your computational resources.'))
        return performance
    
    def set_objective(self, metric):
        """
        Set the metric for optimization. Can also be done by passing to
        `run`.

        Args:
            metric (str): either 'acc' to maximize classification accuracy, or
                    else 'loss' to minimize the loss function
        """
        if metric not in ['loss', 'hvi']:
            raise ValueError(('Invalid metric name {} provided - should be'
                              '"accuracy" or "loss"').format(metric))
        self._metric = metric
        self._objective = "max" if self._metric == "hvi" else "min"
        #TODO currently loss and accuracy
        self._metric_index = 0 
        self._metric_op = METRIC_OPS[self._objective == 'max']
        self._metric_objective = METRIC_OBJECTIVES[self._objective == 'max']

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome[0] = 1
        return genome

    def evaluate_nasbench_genome(self, genome):
        model_nas,level_of_compression,level_of_complexity = self.decode_nasbench(genome)

        result = dict()
        
        #Decode model
        query = self.nasbench_api.query(model_nas)
        result['level_of_compression'] = min(math.log(int(query['trainable_parameters']),10),10)/10

            
        #Initialise performance metrics list
        result['training_time'] = query['training_time']
        result['val_acc'] = query['validation_accuracy']
        result['test_acc'] = query['test_accuracy']
        self.i_model+=1

        return result

    

    
    

        
# Fixed add layer 16.02.2022 
def add_layer(cons, lays, layer,pos):
    # print('> cons before', cons)
    size = len(cons)
    con = int(2**((size)-(pos+1)))
    cons.insert(pos,con)
    lays.insert(pos,layer)
    if(len(cons)>2):
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
    
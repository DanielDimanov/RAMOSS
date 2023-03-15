
import tensorflow as tf
from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
def get_dataset(DATASET=None, batch_size=None, PSS=False):
    multilabel = False
    input_shape = (0,0,0)
    dataset = None
    n_classes = 0
    TRAIN_WITH_GEN = False
    TRAIN_WITH_LOGITS = False
    K.set_image_data_format("channels_last")
    normalize = None
    if(DATASET=='mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        n_classes = 10
        batch_size = 64
    elif(DATASET=='fmnist'):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        n_classes = 10
        batch_size = 64
    elif(DATASET=='cifar10'):
        print('You selected Cifar10')
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32') / 255
        n_classes = 10
        batch_size = batch_size
    elif(DATASET=='oxford_pets'):
        input_dir = "../data/ss/images/"
        target_dir = "../data/ss/annotations/trimaps/"
        img_size = (160, 160)
        num_classes = 3
        if(not batch_size):
            batch_size = 16
        from oxford_pets.oxford_pets import OxfordPets,get_img_paths
        input_img_paths,target_img_paths = get_img_paths(input_dir,target_dir)
        print("Number of samples:", len(input_img_paths))   
        # Split our img paths into a training and a validation set
        val_samples = 1000
        random.Random(1337).shuffle(input_img_paths)
        random.Random(1337).shuffle(target_img_paths)
        train_input_img_paths = input_img_paths[:-val_samples]
        train_target_img_paths = target_img_paths[:-val_samples]
        val_input_img_paths = input_img_paths[-val_samples:]
        val_target_img_paths = target_img_paths[-val_samples:]

        # Instantiate data Sequences for each split
        train_gen = OxfordPets(
            batch_size, img_size, train_input_img_paths, train_target_img_paths
        )
        val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
        dataset = (train_gen,val_gen)
        input_shape = (160,160,3)
        TRAIN_WITH_GEN = True
        n_classes = 3
    elif(DATASET=='city'):

        input_dir = r'C:\data\semantic-segmentation\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\train' #CHANGE THE PATHS!!!
        target_dir = r'C:\data\semantic-segmentation\cityscapes\gtFine_trainvaltest\gtFine\train' #CHANGE THE PATHS!!!
        val_input_dir = r'C:\data\semantic-segmentation\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\val' #CHANGE THE PATHS!!!
        val_target_dir = r'C:\data\semantic-segmentation\cityscapes\gtFine_trainvaltest\gtFine\val' #CHANGE THE PATHS!!!
        test_input_dir = r'C:\data\semantic-segmentation\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\test' #CHANGE THE PATHS!!!
        test_target_dir = r'C:\data\semantic-segmentation\cityscapes\gtFine_trainvaltest\gtFine\test' #CHANGE THE PATHS!!!

        img_size = (512, 1024)
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
        if(PSS):
            from dataloaders.cityscapes.city_pss import City_PSS
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
        input_shape = (320,320,3)
        batch_size = 8
        TRAIN_WITH_GEN = True
        TRAIN_WITH_LOGITS = True
    else:
        raise Exception('Only mnist,fmnist,cifar10, oxford pets and cityscapes are currently supported!')
    
    if(not TRAIN_WITH_GEN):
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        dataset = ((x_train, y_train), (x_test, y_test))
        input_shape = x_train.shape[1:]
    return dataset,input_shape,n_classes, TRAIN_WITH_GEN, TRAIN_WITH_LOGITS, batch_size, normalize, multilabel
    
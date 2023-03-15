import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import time
import random
from dataloaders.cityscapes.cityscapes import Cityscapes,get_img_paths


class City_PSS:
    def __init__(self,input_dir,target_dir,img_size,num_classes,batch_size,NUM_SPLITS):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.NUM_SPLITS = NUM_SPLITS
        input_img_paths,target_img_paths = get_img_paths(input_dir,target_dir)
        print("Number of samples:", len(input_img_paths))   
        # Split our img paths into a training and a validation set
        val_samples = 200
        random.Random(42).shuffle(input_img_paths)
        random.Random(42).shuffle(target_img_paths)
        train_input_img_paths = input_img_paths[:-val_samples]
        train_target_img_paths = target_img_paths[:-val_samples]
        val_input_img_paths = input_img_paths[-val_samples:]
        val_target_img_paths = target_img_paths[-val_samples:]

        self.CITY = Cityscapes(
            batch_size, img_size, train_input_img_paths, train_target_img_paths
        )

        self.LBL_CL_DICT = dict()

        ds_table = pd.DataFrame(list(zip(train_input_img_paths, train_target_img_paths)),
               columns =['Image', 'Target'])
        self.backup_table = ds_table.copy()
        for cl in range(num_classes):
            ds_table[str(cl)] = ds_table.apply(lambda row: self.count_class(row[1],cl,img_size), axis=1)
        self.splits = [set() for _ in range(NUM_SPLITS)]
        for _ in range(int(len(self.backup_table) / (NUM_SPLITS+len(ds_table.columns[2:])))):
            for i_split in range(NUM_SPLITS):
                split = set()
                for col in ds_table.columns[2:]:
                    if(len(ds_table)>0):
                        rec_id = ds_table[ds_table[col] == ds_table[col].max()].index[0]
                        split.add(rec_id)
                        ds_table = ds_table.drop(rec_id)
                    if(len(ds_table==0)):
                        continue
                        rec_id = ds_table[ds_table[col] == ds_table[col].min()].index[0]
                        split.add(rec_id)
                        ds_table = ds_table.drop(rec_id)

                self.splits[i_split] = self.splits[i_split].union(split)

    def count_class(self,lbl_path,cl,img_size):
        if(lbl_path not in self.LBL_CL_DICT.keys()):
            img = load_img(lbl_path, target_size=img_size, color_mode="grayscale")
            lbl_map = np.array(img)
            lbl_map = self.CITY.fix_indxs(lbl_map)
            counts = np.unique(lbl_map, return_counts=True)
            counts_dict = dict(zip(counts[0],counts[1]))
            self.LBL_CL_DICT[lbl_path] = counts_dict
        else:
            counts_dict = self.LBL_CL_DICT[lbl_path]
        if(int(cl) in counts_dict.keys()):
            return counts_dict[int(cl)]
        else:
            return 0

    
    def getSplits(self):
        return [Cityscapes(
            self.batch_size, self.img_size, self.backup_table['Image'].loc[n_split].tolist(), self.backup_table['Target'].loc[n_split].tolist()
        ) for n_split in self.splits]
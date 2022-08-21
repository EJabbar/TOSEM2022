import os
import numpy as np
from random import sample
import tensorflow.keras as keras
from transformers import RobertaTokenizer
import pandas as pd

from dataset_info import Info
from config import Config

class DataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras
    mode: 0 -> TRAIN, !=0 ->VALID
    '''
    def __init__(self, config, list_IDs, labels, dim=(6, 512),
                 n_classes=2, shuffle=True, mode=0, cached=0):
        'Initialization'
        self.dim = dim
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.cached = cached
        self.cache_path = self.config.my_model_cache_train_dataset if mode == 0 else self.config.my_model_cache_valid_dataset 
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.codebert_model)
        self.__load_dataset()
        self.on_epoch_end()

    def __load_dataset(self):
        if self.cached == 0:
            for i, ID in enumerate(self.list_IDs):
                df = pd.read_csv(f'{self.config.processed_path}{ID}.csv')
                df_ret = pd.read_csv(f'{self.config.processed_path}{ID}_ret.csv')
                df['return'] = df['id'].map(df_ret.set_index('id')['return_value'])
                df.fillna('<UKN>', inplace=True)
                mtd_calls = df['called_method'].tolist()
                mtd_calls = '<NXT>'.join(mtd_calls)
                args = df['called_method_args'].tolist()
                args = '<NXT>'.join(args)
                returns = df['return'].to_list()
                returns = '<NXT>'.join(returns)
                mtd_inputs = self.tokenizer.encode_plus(mtd_calls, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
                arg_inputs = self.tokenizer.encode_plus(args, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
                rtn_inputs = self.tokenizer.encode_plus(returns, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)

                tmp = np.array(mtd_inputs['input_ids'] + mtd_inputs['attention_mask'] + arg_inputs['input_ids'] + arg_inputs['attention_mask'] + rtn_inputs['input_ids'] + rtn_inputs['attention_mask'], dtype=np.int32)
                # tmp = tmp.reshape(*self.dim, self.n_channels)
                tmp = tmp.reshape(*self.dim)
                np.save(f'{self.cache_path}{ID}.npy', tmp)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return [X[:,0,:], X[:,1,:], X[:,2,:], X[:,3,:], X[:,4,:], X[:,5,:]], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.int32)
        X = np.empty((self.batch_size, *self.dim), dtype=np.int32)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            tmp = np.load(f'{self.cache_path}{ID}.npy')
            X[i,] = tmp

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



def main():
    list_ids = []
    labels = {}
    info = Info()
    config = Config()

    train_data = [['cli_10', 112, 106, 1],
                 ['cli_11', 120, 114, 1],
                 ['cli_12', 122, 114, 3],
                 ['cli_13', 479, 459, 1],
                 ['cli_14', 481, 461, 1],
                 ['cli_15', 484, 463, 2],
                 ['cli_16', 489, 462, 7],
                 ['cli_17', 145, 138, 1],
                 ['cli_18', 146, 139, 1],
                 ['cli_19', 147, 140, 1],
                 ['cli_20', 148, 141, 1],
                 ['cli_21', 506, 486, 1],
                 ['cli_22', 181, 166, 2],
                 ['cli_23', 173, 159, 2],
                 ['cli_24', 175, 162, 1],
                 ['cli_25', 175, 162, 1],
                 ['cli_26', 188, 175, 1],
                 ['cli_27', 248, 202, 3],
                 ['cli_28', 328, 268, 1],
                 ['cli_29', 340, 280, 1],
    ]

    for version in train_data:
        for bid in range(version[1]):
            list_ids.append(f'{version[0]}_mut_b_{bid}')
            labels[f'{version[0]}_mut_b_{bid}'] = 0
        
        fix_ids = list(range(version[2]))
        for bid in fix_ids:
            list_ids.append(f'{version[0]}_org_f_{bid}')
            labels[f'{version[0]}_org_f_{bid}'] = 1

        for bid in range(version[3]):
            list_ids.append(f'{version[0]}_org_b_{bid}')
            labels[f'{version[0]}_org_b_{bid}'] = 0

    data_generator = DataGenerator(config=config, list_IDs=list_ids, labels=labels, mode=0, cached=1)
    x, l = data_generator.__getitem__(0)
    print(x)

if __name__ == "__main__":
    main()

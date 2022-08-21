import time
import datetime

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import csv
import os
from random import sample
from collections import namedtuple
from typing import List, Iterable, Optional, Dict

import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalMaxPool1D, Bidirectional, LSTM
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import Callback
from pandas import read_csv, DataFrame
from transformers import RobertaConfig, RobertaTokenizer, TFRobertaModel

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from config import Config
from data_generator import DataGenerator
from dataset_info import Info
import logging
import json

logging.basicConfig(level=logging.DEBUG)

ModelOutput = namedtuple('ModelOutput', ['target_index', 'test_vectors'])



class TraceBERTModel():

    def __init__(self, config: Config, info: Info, is_loading=False):
        logging.info('-----------------------Initializing TraceBERT Model... --------------------------')

        self.config = config
        self.info = info
        self.is_loading = is_loading
        self.train_data_generator = None
        self.validation_data_generator = None
        self.predictions: Optional[List] = None
        self.keras_train_model: Optional[keras.Model] = None
        self.keras_model_predict_function: Optional[K.GraphExecutionFunction] = None

        logging.info("****************Initializing data generators*******************")
        self.fetch_data()

        logging.info("****************Creating model*******************")
        self.mtd_codebert = TFRobertaModel.from_pretrained(config.codebert_model)
        self.args_codebert = TFRobertaModel.from_pretrained(config.codebert_model)

        print('-----------------------Initialization Done! --------------------------')


    def fetch_data(self):
        if (self.config.is_training) and (self.is_loading == False):
            list_ids = []
            labels = {}

            for version in self.info.train_data:
                for bid in range(version[1]):
                    list_ids.append(f'{version[0]}_mut_b_{bid}')
                    labels[f'{version[0]}_mut_b_{bid}'] = 0
                
                fix_ids = sample(list(range(version[2])), version[1]+version[3])
                for bid in fix_ids:
                    list_ids.append(f'{version[0]}_org_f_{bid}')
                    labels[f'{version[0]}_org_f_{bid}'] = 1

                for bid in range(version[3]):
                    list_ids.append(f'{version[0]}_org_b_{bid}')
                    labels[f'{version[0]}_org_b_{bid}'] = 0

            ids_train, ids_valid = train_test_split(list_ids, test_size=0.2, random_state=0)

            self.train_data_generator = DataGenerator(config=self.config, list_IDs=ids_train, labels=labels, mode=0)  
            self.validation_data_generator = DataGenerator(config=self.config, list_IDs=ids_valid, labels=labels, mode=1)    

        elif (self.config.is_training) and (self.is_loading == True):

            traces_train = os.listdir(self.config.my_model_cache_train_dataset) 
            traces_train = [t.split('.')[0] for t in traces_train]
            labels_train = {}
            for t in traces_train:
                labels_train[t] = 1 if ('_f_' in t) else 0 

            traces_valid = os.listdir(self.config.my_model_cache_valid_dataset) 
            traces_valid = [t.split('.')[0] for t in traces_valid]
            labels_valid = {}
            for t in traces_valid:
                labels_valid[t] = 1 if ('_f_' in t) else 0 

            self.train_data_generator = DataGenerator(config=self.config, list_IDs=traces_train, labels=labels_train, mode=0, cached=1)  
            self.validation_data_generator = DataGenerator(config=self.config, list_IDs=traces_valid, labels=labels_valid, mode=1, cached=1)   
        

    def _create_keras_model(self):

        # We use another dedicated Keras function to produce predictions.
        # It have additional outputs than the original model.
        # It is based on the trained layers of the original model and uses their weights.
            
        input_mtd_ids_in = Input(shape=(self.config.max_length, ), name='mtd_input_token', dtype='int32')
        input_mtd_masks_in = Input(shape=(self.config.max_length, ), name='mtd_masked_token', dtype='int32') 

        input_arg_ids_in = Input(shape=(self.config.max_length, ), name='arg_input_token', dtype='int32')
        input_arg_masks_in = Input(shape=(self.config.max_length, ), name='arg_masked_token', dtype='int32') 

        # Freeze the BERT model to reuse the pretrained features without modifying them.
        self.mtd_codebert.trainable = False
        self.args_codebert.trainable = False

        mtd_sequence_output = self.mtd_codebert(input_mtd_ids_in, attention_mask=input_mtd_masks_in)
        arg_sequence_output = self.args_codebert(input_arg_ids_in, attention_mask=input_arg_masks_in)

        print(mtd_sequence_output)
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        mtd_bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.config.LSTM_LEN, return_sequences=True)
        )(mtd_sequence_output['last_hidden_state'])

        arg_bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.config.LSTM_LEN, return_sequences=True)
        )(arg_sequence_output['last_hidden_state'])

        # Applying hybrid pooling approach to bi_lstm sequence output.
        mtd_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(mtd_bi_lstm)
        mtd_max_pool = tf.keras.layers.GlobalMaxPooling1D()(mtd_bi_lstm)

        arg_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(arg_bi_lstm)
        arg_max_pool = tf.keras.layers.GlobalMaxPooling1D()(arg_bi_lstm)

        concatenated = tf.keras.layers.concatenate([mtd_avg_pool, mtd_max_pool, arg_avg_pool, arg_max_pool])

        dropout_1 = tf.keras.layers.Dropout(self.config.DROPOUT_RATE)(concatenated)
        test_embedding = tf.keras.layers.Dense(self.config.TEST_VECTOR_SIZE, use_bias=False, activation='tanh')(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(self.config.DROPOUT_RATE)(test_embedding)
        output = tf.keras.layers.Dense(2, activation="softmax")(dropout_2)
        self.keras_train_model = tf.keras.models.Model(
            inputs=[input_mtd_ids_in, input_mtd_masks_in, input_arg_ids_in, input_arg_masks_in], outputs=output
        )
        
        embedding_outputs = ModelOutput(target_index=output, test_vectors=test_embedding)
        self.keras_model_predict_function = K.function(inputs=[input_mtd_ids_in, input_mtd_masks_in, input_arg_ids_in, input_arg_masks_in], outputs=embedding_outputs)
        

    def _create_inner_model(self):
        self._create_keras_model()
        self._compile_keras_model()
        self.keras_train_model.summary()

    def _compile_keras_model(self, optimizer=None):
        if optimizer is None:
            optimizer = self.keras_train_model.optimizer
            if optimizer is None:
                optimizer = self._create_optimizer()

        self.keras_train_model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])


    @classmethod
    def _create_optimizer(cls):
        return keras.optimizers.Adam(lr=5e-5)

    def train(self):
        train_start_time = time.time()
        training_history = self.keras_train_model.fit(self.train_data_generator, 
                            validation_data=self.validation_data_generator, 
			                use_multiprocessing=True,
                            epochs=self.config.epochs, 
                            verbose=self.config.VERBOSE_MODE,
                            workers=-1,                          
                            callbacks=self._create_train_callbacks())

        train_end_time = time.time()
        train_time = str(datetime.timedelta(seconds=int(train_end_time - train_start_time)))
        print(f'Keras Train Time: {train_time}')
        
    
    def fine_tune(self):
        # unfreez the codeBERT layer.
        # recompile the model to make the change effective
        # train end-to-end the model for a few (1, 2, or 3) epochs with unfreezed layers
        # save the model
        print('Start fine-tuning...')
        self.keras_train_model.layers[4].trainable = True
        # self.keras_train_model.layers[5].trainable = True
        self._compile_keras_model(optimizer=keras.optimizers.Adam(lr=1e-5))
        self.keras_train_model.summary()
        
        train_start_time = time.time()
        training_history = self.keras_train_model.fit(self.train_data_generator, 
                            validation_data=self.validation_data_generator, 
			                use_multiprocessing=True,
                            epochs=self.config.fine_tuning_epochs, 
                            verbose=self.config.VERBOSE_MODE,
                            workers=-1,                          
                            callbacks=self._create_train_callbacks())
                            # verbose=self.config.VERBOSE_MODE,     

        train_end_time = time.time()
        train_time = str(datetime.timedelta(seconds=int(train_end_time - train_start_time)))
        print(f'Keras Fine-Tuning Time: {train_time}')


    def plotHistory(self):
        # history = read_csv(self.config.model_checkpoint_csv)
        # # summarize history for accuracy
        # plt.plot(history['accuracy'])
        # plt.plot(history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'Val'], loc='upper left')
        # plt.savefig(os.path.join(self.config.model_dir, 'accuracy_plot'))
        # plt.close()
        # # summarize history for loss
        # plt.plot(history['loss'])
        # plt.plot(history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'Val'], loc='upper left')
        # plt.savefig(os.path.join(self.config.model_dir, 'loss_plot'))
        # plt.close()
        ...

    def evaluate(self):
        # inputs = [self.data, self.mask]
        # outputs = self.encoded_target

        # eval_res = self.keras_train_model.evaluate(inputs, outputs,
        #                                            batch_size=self.config.TRAIN_BATCH_SIZE,
        #                                            verbose=self.config.VERBOSE_MODE)
        # return eval_res
        ...

    def _create_train_callbacks(self):
        keras_callbacks = []
        '''Callback that streams epoch results to a csv file.'''
        csv_log_path = self.config.model_checkpoint_csv
        cb_csv_logger = callbacks.CSVLogger(csv_log_path, append=False, separator=',')
        keras_callbacks.append(cb_csv_logger)
        '''Callback that terminates training when a NaN loss is encountered.'''
        cb_tranining_terminator = callbacks.TerminateOnNaN()
        keras_callbacks.append(cb_tranining_terminator)
        '''Callbacks to save the model at each checkpoint (on each epoch)'''
        file_path = self.config.model_best_weights
        cb_model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path,
                                                        monitor='val_accuracy',
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode='max')
        keras_callbacks.append(cb_model_checkpoint)
        return keras_callbacks


    def predict(self) -> Dict:
        tokenizer = RobertaTokenizer.from_pretrained(self.config.codebert_model)
        for version in self.info.test_data:
            list_ids = []
            labels = {}
            scores = {}
            for bid in range(version[2]):
                list_ids.append(f'{version[0]}_org_f_{bid}')
                labels[f'{version[0]}_org_f_{bid}'] = 1

            for bid in range(version[3]):
                list_ids.append(f'{version[0]}_org_b_{bid}')
                labels[f'{version[0]}_org_b_{bid}'] = 0
            
            predict_start_time = time.time()

            for i, ID in enumerate(list_ids):
                df = pd.read_csv(f'{self.config.processed_path}{ID}.csv')
                mtd_calls = df['called_method'].tolist()
                mtd_calls = '<NXT>'.join(mtd_calls)
                args = df['called_method_args'].tolist()
                args = '<NXT>'.join(args)
                mtd_inputs = tokenizer.encode_plus(mtd_calls, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
                arg_inputs = tokenizer.encode_plus(args, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
                tmp = np.array(mtd_inputs['input_ids'] + mtd_inputs['attention_mask'] + arg_inputs['input_ids'] + arg_inputs['attention_mask'], dtype=np.int32)
                tmp = tmp.reshape(4, 512)
                prediction = self.keras_train_model.predict(([np.array([tmp[0,:]]), np.array([tmp[1,:]]), np.array([tmp[2,:]]), np.array([tmp[3,:]])]))
                scores[ID] = prediction[0][0]

            predict_end_time = time.time()
            predict_time = str(datetime.timedelta(seconds=int(predict_end_time - predict_start_time)))
            print(f'Keras Prediction Time: {predict_time}')

            ranks = sorted(scores, key=scores.get)
            print(ranks)
            for bid in range(version[3]):
                rank = ranks.index(f'{version[0]}_org_b_{bid}')
                print(f'rank {len(ranks) - rank} from {len(ranks)}')
                print(f'norm rank {(len(ranks) - rank)/len(ranks)}')
            print('********************************************************************')


    def rauc_result(self):
        tokenizer = RobertaTokenizer.from_pretrained(self.config.codebert_model)
        for version in self.info.test_data:
            list_ids = []
            labels = {}
            scores = {}
            for bid in range(version[1]):
                list_ids.append(f'{version[0]}_mut_b_{bid}')
                labels[f'{version[0]}_mut_b_{bid}'] = 0

            for bid in range(version[2]):
                list_ids.append(f'{version[0]}_org_f_{bid}')
                labels[f'{version[0]}_org_f_{bid}'] = 1

            for bid in range(version[3]):
                list_ids.append(f'{version[0]}_org_b_{bid}')
                labels[f'{version[0]}_org_b_{bid}'] = 0
            
            for i, ID in enumerate(list_ids):
                df = pd.read_csv(f'{self.config.processed_path}{ID}.csv')
                mtd_calls = df['called_method'].tolist()
                mtd_calls = '<NXT>'.join(mtd_calls)
                args = df['called_method_args'].tolist()
                args = '<NXT>'.join(args)
                mtd_inputs = tokenizer.encode_plus(mtd_calls, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
                arg_inputs = tokenizer.encode_plus(args, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
                tmp = np.array(mtd_inputs['input_ids'] + mtd_inputs['attention_mask'] + arg_inputs['input_ids'] + arg_inputs['attention_mask'], dtype=np.int32)
                tmp = tmp.reshape(4, 512)
                prediction = self.keras_train_model.predict(([np.array([tmp[0,:]]), np.array([tmp[1,:]]), np.array([tmp[2,:]]), np.array([tmp[3,:]])]))
                scores[ID] = prediction[0][0]

            ranks = sorted(scores, key=scores.get)
            print(ranks)
            with open(f'./rauc_{version[0]}.txt', 'w') as convert_file:
                convert_file.write(json.dumps(ranks))

            print('********************************************************************')

    def savePredictions(self):
        # '''
        # :param predictions: the predictions form the model
        # :return: saves the predictions to a csv file for further use.
        # '''
        # print(type(self.config.NUM_SAMPLES))
        # with open(self.config.target_vec_path, 'w') as target_csv, open(self.config.test_vec_path,'w') as test_vec_csv:
        #     target_writer = csv.writer(target_csv, delimiter=',')
        #     test_vec_writer = csv.writer(test_vec_csv, delimiter=',')

        #     for i in range(self.config.NUM_SAMPLES):
        #         row = [self.labels['class names'][i], self.predictions['labels'][i], self.labels['test names'][i]]
        #         target_writer.writerow(row + [self.predictions['target_index'][i]])
        #         test_vec_writer.writerow(row + list(self.predictions['test_vectors'][i]))
        # target_csv.close()
        # test_vec_csv.close()

        # writing test vectors to aggregated result dir for further analysis
        # f_name = os.path.join('output' + '.csv')
        # with open(f_name, 'w') as f:
        #     test_vec_writer = csv.writer(f, delimiter=',')
        #     for i in range(self.config.NUM_SAMPLES):
        #         row = [self.labels['class names'][i], self.predictions['labels'][i], self.labels['test names'][i]]
        #         test_vec_writer.writerow(row + list(self.predictions['test_vectors'][i]))
        # f.close()
        ...

        
    def _load_inner_model(self):
        self._create_inner_model()
        load_path = self.config.model_best_weights
        
        print('Loading model weights from path {}.'.format(load_path))
        self.keras_train_model.load_weights(load_path)
        print('Loading model completed successfully...')
        self.keras_train_model.summary()

    def _save_inner_model(self):
        print('**********************start saving****************************')
        self.keras_train_model.save_weights('/home/ejabbar/emad/Data/Cli/last_weights.h5')
        print(f'Test2Vec model was saved successfully under directory.')


def main():
    ##########################train###########################
    # config = Config()
    # info = Info()
    # t2v = TraceBERTModel(config, info)
    # t2v._create_inner_model()
    # t2v.train()
    ##########################resume training###########################
    # config = Config()
    # info = Info()
    # t2v = TraceBERTModel(config, info,is_loading=True)
    # t2v._load_inner_model()
    # t2v.train()
    ##########################fine-tuning###########################
    # config = Config()
    # info = Info()
    # t2v = TraceBERTModel(config, info,is_loading=True)
    # t2v._load_inner_model()
    # t2v.fine_tune()
    ##########################validation###########################
    config = Config()
    config.is_training = False
    info = Info()
    t2v = TraceBERTModel(config, info,is_loading=False)
    t2v._load_inner_model()
    t2v.predict()

if __name__ == "__main__":
    main()
class Config:
    def __init__(self):
        self.MAX_SEQ = 128
        self.max_length = 512
        self.mutation_raw_data_path = '/home/emad/Desktop/tosem-git/models/dataset/Cli/mutated/'
        self.orig_raw_data_path = '/home/emad/Desktop/tosem-git/models/dataset/Cli/original/'

        self.processed_path = '/home/emad/Desktop/tosem-git/models/dataset/Cli/processed'
        self.my_model_cache_train_dataset = '/home/emad/Desktop/tosem-git/models/dataset/train_cache'
        self.my_model_cache_valid_dataset = '/home/emad/Desktop/models/dataset/valid_cache/'
        self.codebert_model = 'microsoft/codebert-base'
        
        self.is_training = True
        self.DROPOUT_RATE = 0.1
        self.BATCH_SIZE = 8
        self.LSTM_LEN = 50
        self.TEST_VECTOR_SIZE = 100
        self.epochs = 10
        self.fine_tuning_epochs = 5
        self.VERBOSE_MODE = 1
        self.model_checkpoint_csv = "/home/emad/Desktop/models/model/checkpoint_training.log.csv"
        self.model_best_weights = "/home/emad/Desktop/models/model/checkpoint.h5"
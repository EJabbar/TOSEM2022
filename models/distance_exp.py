from itertools import count
from tabnanny import verbose
from config import Config
from dataset_info import Info
from transformers import RobertaTokenizer
from model import TraceBERTModel
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

config = Config()
config.is_training = False
info = Info()

tokenizer = RobertaTokenizer.from_pretrained(config.codebert_model)
########################load model##############
t2v = TraceBERTModel(config, info,is_loading=False)
t2v._load_inner_model()

predict_func = t2v.keras_model_predict_function
########################format mtd##############

def generate_input(df):   
    mtd_calls = df['called_method'].tolist()
    mtd_calls = '<NXT>'.join(mtd_calls)
    args = df['called_method_args'].tolist()
    args = '<NXT>'.join(args)
    mtd_inputs = tokenizer.encode_plus(mtd_calls, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
    arg_inputs = tokenizer.encode_plus(args, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
    tmp = np.array(mtd_inputs['input_ids'] + mtd_inputs['attention_mask'] + arg_inputs['input_ids'] + arg_inputs['attention_mask'], dtype=np.int32)
    tmp = tmp.reshape(4, 512)
    return tmp


########################load data###############
import pandas as pd
import numpy as np


for project in info.test_data:
    id_vector = {}
    failing_path = f'{config.processed_path}/{project[0]}_org_b_0.csv'
    failing_trace = pd.read_csv(failing_path)
    failing_suit = failing_trace.iloc[0][2]
    print(failing_suit)
    tmp = generate_input(failing_trace)
    id_vector['b_0'] = predict_func(([np.array([tmp[0,:]]), np.array([tmp[1,:]]), np.array([tmp[2,:]]), np.array([tmp[3,:]])]))[1][0].tolist()
    num_tests = 0
    for i in range(project[2]):
        trc_path = f'{config.processed_path}/{project[0]}_org_f_{i}.csv'
        trc = pd.read_csv(trc_path)
        try:
            trc_suit = trc.iloc[0][2]
        except:
            trc_suit = ''

        if trc_suit == failing_suit:
            num_tests += 1
            tmp = generate_input(trc)
            id_vector[f'f_{i}'] = predict_func(([np.array([tmp[0,:]]), np.array([tmp[1,:]]), np.array([tmp[2,:]]), np.array([tmp[3,:]])]))[1][0].tolist()

    print('number of tests: ', num_tests)

    vectors = np.array(list(id_vector.values()))
    centroid = vectors.mean(axis=0)

    id_dist = {}
    vectors = []
    for key, value in id_vector.items():
        vectors.append(value)
        id_dist[key]=np.linalg.norm(value-centroid)

    sorted = {k: v for k, v in sorted(id_dist.items(), key=lambda item: item[1])}

    keys = list(id_vector.keys())
    print(keys)
    list_ranks = [0]*len(vectors)
    c = 1
    for key, value in sorted.items():
        list_ranks[keys.index(key)] = 87-c
        print(key, '->', value)
        c += 1

    print(list_ranks)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    np.random.seed(42)
    sns.set(font_scale=2)

    vectors = np.array(vectors)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', vectors.shape)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(vectors)

    labels = ['Passing']* len(tsne_results[:,1])
    labels[0] = 'Failing'
    cols = [80]* len(tsne_results[:,1])
    cols[0] = 10
    print(labels)


    plt.figure(figsize=(16,10))
    swarm_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        c=cols,
        hue=labels,
        style=labels,
        legend="full",
        alpha=1,
        markers=["X","o"],
        s=180
    )

    # for i in range(len(list_ranks)):
    #     if list_ranks[i] < 3:
    #         plt.text(tsne_results[i,0]-1, tsne_results[i,1], list_ranks[i], horizontalalignment='left', size='medium', color='black', weight='semibold')

    fig = swarm_plot.get_figure()
    fig.savefig("./out_tmp.png") 
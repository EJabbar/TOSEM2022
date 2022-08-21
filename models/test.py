# import pandas as pd
# pd.set_option('display.max_columns', 7)


# df2 = pd.read_csv('/home/emad/Desktop/tosem-git/models/dataset/Cli/processedcli_10_mut_b_1_ret.csv')

# df = pd.read_csv('/home/emad/Desktop/tosem-git/models/dataset/Cli/processedcli_10_mut_b_1.csv')


# print(df)
# print(df2)

# # df['return'] = df2['return_value'].where(df['id'].isin(df2['id']))
# df['return'] = df['id'].map(df2.set_index('id')['return_value'])


# print(df)
# mtd_calls = df['called_method'].tolist()
# mtd_calls = '<NXT>'.join(mtd_calls)
# args = df['called_method_args'].tolist()
# args = '<NXT>'.join(args)
# mtd_inputs = self.tokenizer.encode_plus(mtd_calls, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
# arg_inputs = self.tokenizer.encode_plus(args, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
# tmp = np.array(mtd_inputs['input_ids'] + mtd_inputs['attention_mask'] + arg_inputs['input_ids'] + arg_inputs['attention_mask'], dtype=np.int32)
# # tmp = tmp.reshape(*self.dim, self.n_channels)
# tmp = tmp.reshape(*self.dim)
# np.save(f'{self.cache_path}{ID}.npy', tmp)

import numpy as np
from sklearn.neighbors._kde import KernelDensity

### create data ###
sample_count = 20
n = 512
data = np.random.randn(sample_count, n)
data_norm = np.sqrt(np.sum(data*data, axis=1))
data = data/data_norm[:, None]   # Normalized data to be on unit sphere


## estimate pdf using KDE with gaussian kernel
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)

lop_p = kde.score_samples(data)  # returns log(p) of data sample
p = np.exp(lop_p)                # estimate p of data sample
entropy = -np.sum(p*lop_p) 
print(entropy)
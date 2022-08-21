import logging
from numpy import argmax
import pandas as pd
import math
import os

from config import Config

logging.basicConfig(level=logging.DEBUG)

"""
1- remove repeatative calls
2- remove junit calls
3- force max sequence of calls (cut from middel)
4- abstract values
"""

class DataProcessor():

    def __init__(self, config, project, version, mut_trc, b_trace):
        self.config = config
        self.project = project
        self.version = version
        self.mut_trc = mut_trc
        self.b_trace = b_trace


    def process(self):
        logging.info(f"start preprocessing for {self.project}.")
        try:
            for mt in self.mut_trc:
                self.read_file(mt[0]+mt[1], mt[0]+mt[2])
                self.remove_junit_calls()
                self.remove_rep_calls()
                self.force_max_calls()
                self.filter_ret()

                self.abstract_values()
                self.write_to_file(self.config.processed_path+f'{self.project}_{self.version}_mut_{mt[1]}', self.config.processed_path+f'{self.project}_{self.version}_mut_{mt[2]}')

            for bt in self.b_trace:
                self.read_file(bt[0]+bt[1], bt[0]+bt[2])
                self.remove_junit_calls()
                self.remove_rep_calls()
                self.force_max_calls()
                self.filter_ret()
                self.abstract_values()
                self.write_to_file(self.config.processed_path+f'{self.project}_{self.version}_org_{bt[1]}', self.config.processed_path+f'{self.project}_{self.version}_org_{bt[2]}')
        except:
            self.df.to_csv('./err.csv')
            raise


    def read_file(self, pth, pth_ret):
        self.df = pd.read_csv(pth)
        self.df = self.df.iloc[: , 1:]
        self.df.dropna(subset = ["id", "calling_class", "called_class"], inplace=True)
        self.df = self.df.loc[self.df.id.apply(type) == int]

        self.df_ret = pd.read_csv(pth_ret)
        self.df_ret = self.df_ret.iloc[: , 1:]
        self.df_ret.dropna(subset = ["id", "return_type", "return_value"], inplace=True)
        self.df_ret = self.df_ret.loc[self.df_ret.id.apply(type) == int]



    def remove_rep_calls(self):
        self.df = self.df.reset_index(drop=True)
        prev_called_method = ''
        prev_called_args = ''
        removed_indces = []
        for index, row in self.df.iterrows():
            if (row['called_method'] == prev_called_method) and (row['called_method_args'] == prev_called_args):
                removed_indces.append(index)

            prev_called_method = row['called_method']
            prev_called_args = row['called_method_args']

        logging.info(f"drop {len(removed_indces)} reptative rows")
        self.df = self.df.drop(self.df.index[removed_indces])

    def remove_junit_calls(self):
        self.df = self.df.loc[(~self.df['calling_class'].str.contains("junit")) & (~self.df['called_class'].str.contains("junit"))]
        
    def force_max_calls(self):
        if len(self.df) > self.config.MAX_SEQ:
            logging.info(f'cut dataset with len: {len(self.df)}')
            half_max = math.ceil(self.config.MAX_SEQ/2)
            self.df = self.df.reset_index(drop=True)
            self.df = self.df.loc[(self.df.index < half_max) | (self.df.index >= len(self.df)-half_max)]
        
        self.df = self.df.reset_index(drop=True)

    def filter_ret(self):
        ids = self.df['id'].to_list()
        self.df_ret = self.df_ret[self.df_ret['id'].isin(ids)]

    def abstract_values(self):
        '''
        String:
        <EMPATY_STRING>, <STRING>

        Int:
        <ZERO>, <SMALL_NEG>, <NEG>, <LARGE_NEG>, <SMALL_POS>, <POS>, <LARGE_POS>

        Bool: True, False

        Float:
        <ZERO>, <SMALL_NEG>, <NEG>, <LARGE_NEG>, <SMALL_POS>, <POS>, <LARGE_POS>

        Object: refrence
        '''

        def map_values(tv):
            tp = tv.split(':')[0]
            val = tv.split(':')[1]
            if tp == "String" or tp == "StringBuffer":
                if val == "":
                    return f'{tp}:ES'
                else:
                    return f'{tp}:STR'

            elif tp in ["Float", "Double", "Integer", "float", "int", "double", "short"]:
                try:
                    val = float(val)
                    if val < -10000:
                        return f'{tp}:LNV'
                    elif val < 0:
                        return f'{tp}:NV'
                    elif val == 0:
                        return f'{tp}:ZERO'
                    elif val < 10000:
                        return f'{tp}:PV'
                    else:
                        return f'{tp}:LPV'
                except ValueError:
                    return f'{tp}:ERR'

            elif '[' in tp:
                if len(val) == 2:
                    return f'{tp}:EL'
                elif ',' in val:
                    return f'{tp}:ML'
                else:
                    return f'{tp}:OL'

            else:
                return tv

        for index, row in self.df.iterrows():
            args = row['called_method_args']
            if ('<SP>' not in args) or ('<EP>' not in args):
                continue
            args = args[4:-4]
            args = args.split('<NP>')
            while '' in args: args.remove('')    
            args = list(map(map_values, args))
            if len(args) > 0:
                args = '<NP>'.join(args)
            else:
                args = ''
            self.df.at[index, 'called_method_args'] = f'<SP>{args}<EP>'

        for index, row in self.df_ret.iterrows():
            rv = '<UK>'
            if row['return_value'] != ' ' :
                rv=row['return_value']
            args = row['return_type']+':'+rv  
            args = map_values(args)
            self.df_ret.at[index, 'return_value'] = f'{args}'

    def write_to_file(self, pth, pth_rtn):
        self.df.to_csv(pth)
        self.df_ret.to_csv(pth_rtn)

    


def main():
    c = Config()
    project = 'cli'
    versions = list(range(10, 30))
    for version in versions:
        list_mutation_traces = os.listdir(f'{c.mutation_raw_data_path}{project}_{version}f/') 
        list_mutation_traces = [[f'{c.mutation_raw_data_path}{project}_{version}f/', i, i[:-4]+'_ret.csv'] for i in list_mutation_traces]
        print(list_mutation_traces)
        list_mutation_traces = [i for i in list_mutation_traces if ('ret' not in i[1])]
        list_bversion_traces = os.listdir(f'{c.orig_raw_data_path}{project}_{version}b/')
        list_bversion_traces = [[f'{c.orig_raw_data_path}{project}_{version}b/', i, i[:-4]+'_ret.csv'] for i in list_bversion_traces]
        list_bversion_traces = [i for i in list_bversion_traces if ('ret' not in i[1])]
        d = DataProcessor(c, project, version, list_mutation_traces, list_bversion_traces)
        d.process()


if __name__ == "__main__":
    main()

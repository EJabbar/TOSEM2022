import re
import random
import pandas as pd
import logging
import sys
from io import StringIO
import re
from reader import read_rtn, read_tst

random.seed(0)

project = sys.argv[1].lower()
version = sys.argv[2]


pattern = re.compile("^[0-9]+\)(.*)\((.*)\)$")
file_path = f'./trace4j/{project}_{version}_script_log.txt'
def clean_failing_tests(s):
    s = s.strip()
    s = s.split(' ')[1]
    return s

def split_clss_tstname(s):
    sl = s.split("(")
    name = sl[0]
    clss = sl[1][:-1]
    return [clss, name]

with open(file_path) as log:
    failing_tests = list(filter(pattern.match, log.readlines()))
    failing_tests = list(map(clean_failing_tests, failing_tests))
    failing_tests = list(map(split_clss_tstname, failing_tests))
    failing_tests = list(filter(lambda f: 'junit.framework' not in f[0], failing_tests))

failing_classes= [i[0] for i in failing_tests]
failing_names = [i[1] for i in failing_tests]

list_tests = []

with open(f'./trace4j/projects/{project}_{version}/all_tests') as f:
    lines = f.readlines()
    for line in lines:
        if line != "":
            list_tests.append(line)

def clean_tests(s):
    s = s.strip().split('(')
    tn = s[0]
    cn = s[1].split(')')[0]
    return [cn, tn]

list_tests = list(map(clean_tests, list_tests))

print(list_tests)

warning_counter = 0
test_number = 0

fail_counter = 0

for tst in list_tests:
    tst_path = f'./trace4j/projects/{project}_{version}/trace4j/logs/log_{tst[0]}.csv'

    try:
        df = read_tst(tst_path)
    except:
        print(f'cannot read file: {tst_path}')
        continue

    if len(df[(df.calling_method==tst[1])]) == 0:
        logging.warning(f'test not found in module for testcase: {tst}')
        warning_counter+=1 
        continue

    start_idx = df[(df.calling_method==tst[1])].index[0]
    try:
        end_idx = df[(df.calling_class=='junit.framework.TestResult') & (df.calling_method=='endTest') & (df.index > start_idx)].index[0]
    except:
        try:
            end_idx = df[((df.calling_class.str.contains('junit')) | (df.called_class.str.contains('junit'))) & (df.index > start_idx)].index[0]
        except:
            end_idx = df.index[-1]

    df = df.truncate(before=start_idx, after=end_idx)

    if (tst[0] in failing_classes) and (tst[1] in failing_names):
        df.to_csv(f'./dataset/{project}_{version}/b_{fail_counter}.csv')
        ids = df['id'].to_list()
        rtn_df = read_rtn(f'./trace4j/projects/{project}_{version}/trace4j/logs/ret_log_{tst[0]}.csv')
        rtn_df = rtn_df[rtn_df['id'].isin(ids)]
        rtn_df.to_csv(f'./dataset/{project}_{version}/b_{fail_counter}_ret.csv')
        fail_counter+=1
    else:
        df.to_csv(f'./dataset/{project}_{version}/f_{test_number}.csv')
        ids = df['id'].to_list()
        rtn_df = read_rtn(f'./trace4j/projects/{project}_{version}/trace4j/logs/ret_log_{tst[0]}.csv')
        rtn_df = rtn_df[rtn_df['id'].isin(ids)]
        rtn_df.to_csv(f'./dataset/{project}_{version}/f_{test_number}_ret.csv')
        test_number+=1



print(f'from {test_number} tests, {warning_counter} has warning')


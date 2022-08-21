import re
import random
import pandas as pd
import logging
import sys
from io import StringIO
import re

random.seed(0)

project = sys.argv[1]
version = sys.argv[2]

num_mutants = 0

with open(f'./tmp/num_tests_{project}_{version}.txt') as f:
    ln = f.readline()
    num_mutants = int(ln.split()[0])

project = project.lower()
warning_counter = 0
mutant_number = 0

for mv in range(num_mutants):
    print(mv)
    pattern = re.compile("^[0-9]+\)(.*)\((.*)\)$")

    file_path = f'./trace4j/logs/{project}_{version}_{mv}_script_log.txt'

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

    if len(failing_tests) == 0:
        logging.warning(f'mutant {mv} has no failing test')
        warning_counter+=1 
        continue

    selected_failing = random.choice(failing_tests)
    selected_pkg = selected_failing[0].rsplit(".", 1) [0]

    tst_path = f'./trace4j/projects/{project}_{version}_{mv}/trace4j/logs/log_{selected_failing[0]}.csv'

    df = pd.read_csv(tst_path, usecols=range(6))

    if len(df[(df.calling_method==selected_failing[1])]) == 0:
        logging.warning(f'test not found in module for mutant {mv}')
        warning_counter+=1 
        continue

    start_idx = df[(df.calling_method==selected_failing[1])].index[0]
    try:
        end_idx = df[(df.calling_class=='junit.framework.TestResult') & (df.calling_method=='endTest') & (df.index > start_idx)].index[0]
    except:
        try:
            end_idx = df[((df.calling_class.str.contains('junit')) | (df.called_class.str.contains('junit'))) & (df.index > start_idx)].index[0]
        except:
            end_idx = df.index[-1]
            
    df = df.truncate(before=start_idx, after=end_idx)
    df.to_csv(f'./dataset/{project}_{version}/b_{mutant_number}.csv')
    mutant_number+=1

print(f'from {num_mutants} mutants, {warning_counter} has warning')

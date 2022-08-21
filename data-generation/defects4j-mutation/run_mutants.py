from csv import reader
import sys
import os
import shutil
import subprocess
import random
import pandas as pd
import logging
from io import StringIO
import re
import signal
from tqdm import tqdm
from reader import read_tst, read_rtn

random.seed(10)

mutants_dir = sys.argv[1]
project = sys.argv[2]
version = sys.argv[3]

lower_project = project.lower()
warning_counter = 0
mutant_number = 0

num_tests = 0

os.system(f'mkdir ./dataset/{lower_project}_{version}')

with open(f'./tmp/num_tests_{project}_{version}.txt') as f:
    ln = f.readline()
    num_tests = int(ln.split()[0])

mutants = os.listdir(mutants_dir)

random.shuffle(mutants)

# m_path = mutants_dir + mutants[0]
# shutil.copytree(m_path, project_path+"/src/java/", dirs_exist_ok=True)
i = 0
generated_traces = 0 
while (generated_traces < num_tests) and (i < len(mutants)-1):
    m = mutants[i]
    project_path = f'./trace4j/projects/{project}_{version}_{i}/'.lower()
    os.system(f'cp -R ./tmp/{project}_{version}_orig/. {project_path}')
    m_path = mutants_dir + m
    if os.path.isdir(project_path+"/src/main/java/"):
        shutil.copytree(m_path, project_path+"/src/main/java/", dirs_exist_ok=True)
    elif os.path.isdir(project_path+"/src/java/"):
        shutil.copytree(m_path, project_path+"/src/java/", dirs_exist_ok=True)
    else:
        print("Warning: folders not match in project!!!")


    try:
        p = subprocess.Popen(f'cd trace4j ; bash run.sh {project} {version} {i} > /dev/null 2>&1', start_new_session=True, shell=True)
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        
        # insert code here
        print(f'mutant: {m} skipped')
        subprocess.call(f'rm -rf ./trace4j/projects/{lower_project}_{version}_{i}/' , timeout=150, shell=True)
        i += 1
        continue

    #########################################################################
    pattern = re.compile("^\s+-\s+.*::.*$") # re.compile("^[0-9]+\)(.*)\((.*)\)$")

    file_path = f'./trace4j/logs/{lower_project}_{version}_{i}_script_log.txt'


    def clean_failing_tests(s):
        s = s.strip()
        s = s[2:]
        return s

    def split_clss_tstname(s):
        sl = s.split("::")
        clss = sl[0]
        name = sl[1]
        return [clss, name]
    # def clean_failing_tests(s):
    #     s = s.strip()
    #     s = s.split(' ')[1]
    #     return s

    # def split_clss_tstname(s):
    #     sl = s.split("(")
    #     name = sl[0]
    #     clss = sl[1][:-1]
    #     return [clss, name]

    subprocess.call(f'mv ./trace4j/{lower_project}_{version}_{i}_script_log.txt ./trace4j/logs/', timeout=150, shell=True)

    with open(file_path) as log:
        failing_tests = list(filter(pattern.match, log.readlines()))
        failing_tests = list(map(clean_failing_tests, failing_tests))
        failing_tests = list(map(split_clss_tstname, failing_tests))
        failing_tests = list(filter(lambda f: 'junit.framework' not in f[0], failing_tests))

    if len(failing_tests) == 0:
        logging.warning(f'mutant {i} has no failing test')
        warning_counter+=1 
        subprocess.call(f'rm -rf ./trace4j/projects/{lower_project}_{version}_{i}/' , timeout=150, shell=True)
        i += 1
        continue

    with open(f'./trace4j/projects/{lower_project}_{version}_{i}/trace4j/test_statistic/test_all.txt') as f:
        ssuite = f.readline().strip()
    
    failing_tests = list(filter(lambda f: ssuite in f[0], failing_tests))

    if len(failing_tests) == 0:
        logging.warning(f'mutant {i} has no failing test')
        warning_counter+=1 
        subprocess.call(f'rm -rf ./trace4j/projects/{lower_project}_{version}_{i}/' , timeout=150, shell=True)
        i += 1
        continue

    selected_failing = random.choice(failing_tests)
    selected_pkg = selected_failing[0].rsplit(".", 1) [0]

    tst_path = f'./trace4j/projects/{lower_project}_{version}_{i}/trace4j/logs/log_{selected_failing[0]}.csv'

    df = read_tst(tst_path)

    if len(df[(df.calling_method==selected_failing[1])]) == 0:
        logging.warning(f'test not found in module for mutant {i}')
        warning_counter+=1 
        subprocess.call(f'rm -rf ./trace4j/projects/{lower_project}_{version}_{i}/' , timeout=150, shell=True)
        i += 1
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
    df.to_csv(f'./dataset/{lower_project}_{version}/b_{mutant_number}.csv')

    ids = df['id'].to_list()

    rtn_df = read_rtn(f'./trace4j/projects/{lower_project}_{version}_{i}/trace4j/logs/ret_log_{selected_failing[0]}.csv')
    rtn_df = rtn_df[rtn_df['id'].isin(ids)]
    rtn_df.to_csv(f'./dataset/{lower_project}_{version}/b_{mutant_number}_ret.csv')
    mutant_number+=1
    subprocess.call(f'rm -rf ./trace4j/projects/{lower_project}_{version}_{i}/' , timeout=150, shell=True)
    i += 1
    generated_traces += 1
    print(f'failing {mutant_number} from {num_tests} failings generated')

   
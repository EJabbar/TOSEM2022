from csv import reader
from pandas.core.common import flatten
import xml.etree.ElementTree as ET
from random import choice
import random
from statistics import mean
import pandas as pd
from tqdm import tqdm
import os

random.seed(12)

def get_covered_lines(file_path):
    class_cvlines = {}
    if not os.path.exists(file_path):
        return class_cvlines
    tree = ET.parse(file_path)
    root = tree.getroot()
    for pkgs in root.findall('packages'):
        for pkg in pkgs.findall('package'):
            for clz in pkg.findall('classes'):
                for cls in clz.findall('class'):
                    cl_name = cls.get('name')
                    class_cvlines[cl_name] = {}
                    for ls in cls.findall('lines'):
                        for l in ls.findall('line'):
                            l_n = int(l.get('number'))
                            l_h = int(l.get('hits'))
                            class_cvlines[cl_name][l_n]= l_h
    return class_cvlines

def BC(project, version):

    num_of_test = 0
    tests_in_failing_suite = []
    failing_tests = []
    with open(f'./coverage_results/{project}/{project}_{version}_tests.txt', "r") as f:
        nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
        tests_in_failing_suite = nonempty_lines
        num_of_test = len(nonempty_lines)
        f.close()

    with open(f'./failing_tests/{project}/failing_{version}.txt', "r") as f:
        ftests = [line.strip("\n") for line in f if line != "\n"]
        f.close()

    for ft in ftests:
        i = tests_in_failing_suite.index(ft) if ft in tests_in_failing_suite else -1
        if i != -1:
            failing_tests.append(i)

    def exec():
        version_results = []
        
        types = []
        covered_branches = []
        lines_of_branches = []
        classes = []

        def get_branches():
            with open('./branches/{}/{}_{}.csv'.format(project, project, version), 'r') as brchcsv:
                csv_reader = reader(brchcsv)
                for row in csv_reader:
                    covered_branches.append([False]*(len(row)-2))
                    types.append(row[1])
                    classes.append(row[0])
                    lines_of_branches.append([int(i) for i in row[2:]])

        get_branches()

        def get_score_of_test(tstcvrdlines):
            score = 0
            for i in range(len(lines_of_branches)):
                clsname = classes[i]
                lines = lines_of_branches[i]
                tp = types[i]
                if tp == 'jump':
                    h_cnd = 0
                    h_nxt = 0
                    if clsname in tstcvrdlines.keys():
                        h_cnd = tstcvrdlines[clsname][lines[0]]
                        h_nxt = tstcvrdlines[clsname][lines[1]]
                    brnch_false = (h_cnd-h_nxt) > 0
                    brnch_true = h_nxt > 0
                    if brnch_false and covered_branches[i][0] == False:
                        score = score + 1
                    if brnch_true and covered_branches[i][1] == False:
                        score = score + 1
                elif tp == 'switch':
                    for j, lbr in enumerate(lines):
                        if clsname in tstcvrdlines.keys():
                            if tstcvrdlines[clsname][lbr] > 0 and covered_branches[i][j] == False:
                                score == score + 1
            return score

        test_covered = {}
        for i in range(num_of_test):
            file_path = './coverage_results/coverage_{}_{}.xml'.format(version, i)
            cl = get_covered_lines(file_path)
            test_covered[i] = cl

        rank = []
        remained_tests = list(range(num_of_test))

        def rank_remained_tests():
            id_score = {}
            for tst in remained_tests:
                score = get_score_of_test(test_covered[tst])
                id_score[tst] = score
            return id_score

        def add_test_to_covered_branches(tstcvrdlines):
            for i in range(len(lines_of_branches)):
                clsname = classes[i]
                lines = lines_of_branches[i]
                tp = types[i]
                if tp == 'jump':
                    h_cnd = 0
                    h_nxt = 0
                    if clsname in tstcvrdlines.keys():
                        h_cnd = tstcvrdlines[clsname][lines[0]]
                        h_nxt = tstcvrdlines[clsname][lines[1]]
                    brnch_false = (h_cnd-h_nxt) > 0
                    brnch_true = h_nxt > 0
                    if brnch_false:
                        covered_branches[i][0] = True
                    if brnch_true:
                        covered_branches[i][1] = True
                elif tp == 'switch':
                    for j, lbr in enumerate(lines):
                        if clsname in tstcvrdlines.keys():
                            if tstcvrdlines[clsname][lbr] > 0:
                                covered_branches[i][j] = True

        while len(remained_tests) > 0:
            id_score = rank_remained_tests()
            
            vls = id_score.values()
            mx_vl = max(vls)
            mx_ks = [k for k, v in id_score.items() if v == mx_vl]
            nid = choice(mx_ks)
            add_test_to_covered_branches(test_covered[nid])
            rank.append(nid)
            remained_tests.remove(nid)

        #print('----------------version: {} ---------------------'.format(version))
        #print('num of tests: ', num_of_test)
        frl = []
        for ft in failing_tests:
            frl.append(rank.index(ft)+1)

        version_results.append(min(frl))

        return version_results

    final_rslts = []
    print(failing_tests)
    for i in tqdm(range(0, 30)):
        rslt = exec()
        final_rslts.append(rslt[0]/num_of_test)

    with open('./results_BC.txt', 'a+') as f:
        f.write(f'BC average FFR for {project} version {version}: {mean(final_rslts)}\n')

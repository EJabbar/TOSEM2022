from xml.etree.ElementTree import VERSION

from parse import get_covered_lines
from random import choice
import pandas as pd
from tqdm import tqdm
from statistics import mean
import random

random.seed(12)


def LC(project, version):

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

        test_covered = {}
        for i in range(num_of_test):
            file_path = './coverage_results/{}/coverage_{}_{}.xml'.format(project, version, i)
            cl = get_covered_lines(file_path)
            test_covered[i] = cl

        rank = []

        # a = covered_lines['ar']
        covered_lines = {}
        remained_tests = list(range(num_of_test))

        def get_score(lines):
            new_lines = 0
            for k in lines.keys():
                covered_set = set([])
                if k in covered_lines.keys():
                    covered_set = set(covered_lines[k])
                new_covered_set = set(lines[k])
                size_new_covered = len(new_covered_set.difference(covered_set))
                new_lines += size_new_covered

            return new_lines

        def rank_remained_tests():
            id_score = {}
            for tst in remained_tests:
                score = get_score(test_covered[tst])
                id_score[tst] = score
            
            return id_score


        while len(remained_tests) > 0:
            id_score = rank_remained_tests()
            
            vls = id_score.values()
            mx_vl = max(vls)
            mx_ks = [k for k, v in id_score.items() if v == mx_vl]
            nid = choice(mx_ks)
            
            for k in test_covered[nid].keys():
                if k not in covered_lines.keys():
                    covered_lines[k] = set([])
                covered_lines[k].update(set(test_covered[nid][k]))
            rank.append(nid)
            remained_tests.remove(nid)

        #print('----------------version: {} ---------------------'.format(version))
        #print('num of tests: ', num_of_test)
        frl = []
        for ft in failing_tests:
            frl.append(rank.index(ft)+1)

        #print('FFT: ', min(frl))

        version_results.append(min(frl))

        return version_results

    final_rslts = []
    print(failing_tests)
    for i in tqdm(range(0, 30)):
        rslt = exec()
        final_rslts.append(rslt[0]/num_of_test)

    with open('./results_LC.txt', 'a+') as f:
        f.write(f'LC average FFR for {project} version {version}: {mean(final_rslts)}\n')

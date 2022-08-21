from logging.config import valid_ident
import subprocess
from tqdm import tqdm


def get_coverage(pname, v):

    subprocess.run("bash ./clone.sh {} {}".format(pname, v), shell=True)
    subprocess.run("bash ./relevant.sh {} {}".format(pname, v), shell=True)
    subprocess.run("bash ./failing_tests.sh {} {}".format(pname, v), shell=True)

    def reformat_testcase(tst):
        tmp = tst.split('(')
        name = tmp[0]
        tmp = tmp[1].split(')')
        path = tmp[0]
        return path+'::'+name




    rel_file_path = "./relevant_tests/{}/relevant_{}.txt".format(pname, v)
    tests = []
    with open(rel_file_path, 'r') as rf:
        tests = rf.read().splitlines()
    tests = list(map(reformat_testcase, tests))
    print('##############################################')
    fail_file_path = "./failing_tests/{}/failing_{}.txt".format(pname, v)
    with open(fail_file_path, 'r') as rf:
        fail_test = rf.read().splitlines()[0]
    failing_class =  fail_test.split('::')[0]
    fsuite_tsts = list(filter(lambda t: failing_class in t, tests))

    with open(f'./coverage_results/{pname}/{pname}_{v}_tests.txt', 'w') as tstsfile:
        for t in fsuite_tsts:
            tstsfile.write(t + "\n")
        tstsfile.close()   


    for ti, t in tqdm(enumerate(fsuite_tsts), total=len(fsuite_tsts), leave=False):
        rc = subprocess.run("bash ./get_coverage.sh {} {} {} {}".format(v, ti, t, pname), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

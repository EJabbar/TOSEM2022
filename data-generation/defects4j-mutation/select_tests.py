import sys
import os
import shutil
import random

# random.seed(10)

# project_dir = sys.argv[1]
# project = sys.argv[2]
# version = sys.argv[3]

# num_tests = 0

# with open(f'./tmp/num_tests_{project}_{version}.txt') as f:
#     ln = f.readline()
#     num_tests = int(ln.split()[0])

# num_mutants = len(os.listdir(project_dir+'mutants/'))
# selected_mutants = random.sample(range(1, num_mutants+1), num_tests)

# for i in range(1, num_mutants+1):
#     if i not in selected_mutants:
#         shutil.rmtree(project_dir+'mutants/'+str(i))




import sys
import re
import random

project_path = sys.argv[1]

print(project_path)

failing_suits = []
with open(f'{project_path}/failing_tests') as f:
    for line in f.readlines():
        if re.findall("^---", line):
            tmp = line.split(' ')[1]
            tmp = tmp.split(':')[0]
            tmp = tmp.strip()
            failing_suits.append(tmp)

failing_suits = list(set(failing_suits))

random_selected_failing = []
if len(failing_suits) > 0:
    random_selected_failing.append(random.choice(failing_suits))

with open(f'{project_path}/trace4j/test_statistic/test_all.txt', 'w') as test_all:
    for fs in random_selected_failing:
        test_all.write(fs + '\n')

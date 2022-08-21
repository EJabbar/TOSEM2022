rng = 66
pname = 'Lang'
expt = 2


def reformat_testcase(tst):
    tmp = tst.split('(')
    name = tmp[0]
    tmp = tmp[1].split(')')
    path = tmp[0]
    return path+'::'+name

num = []
fids = []
for i in range(1, rng):
    if i == expt:
        num.append(0)
        fids.append([])
    else:
        pth = './relevant_tests/relevant_{}.txt'.format(i)
        tests = []
        with open(pth, 'r') as f:
            tests = f.read().splitlines()
            num.append(len(tests))
            tests = list(map(reformat_testcase, tests))
        
        ftests = []
        pth = './failing_tests/failing_{}.txt'.format(i)
        with open(pth, 'r') as f:
            ftests = f.read().splitlines()
        
        ftestsid = []
        for ft in ftests:
            fid = tests.index(ft)
            ftestsid.append(fid)
        fids.append(ftestsid)


print(num)
print(fids)

from numpy import ma
import pandas as pd
import re

def follow_pattern(s):
    pattern = re.compile("^(.*?),\"(.*?)\",\"(.*?)\",\"(.*?)\",\"(.*?)\",(.*?)", re.DOTALL)
    return True if pattern.match(s) else False


def split_line(s):
    pattern = re.compile("(.*?),\"(.*?)\",\"(.*?)\",\"(.*?)\",\"(.*?)\",\"(.*<EP>?)\"", re.DOTALL)
    m = pattern.match(s)
    return [m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)]


def read_tst(tst_path):

    with open(tst_path) as file:
        lines = file.readlines()[1:]
        is_line = list(map(follow_pattern, lines))
        nlines = []
        for i, l in enumerate(lines):
            if is_line[i]:
                nlines.append(l)
            else:
                nlines[-1] = nlines[-1] + l
        nlines = list(map(split_line, nlines))
        df = pd.DataFrame(nlines, columns=['id','calling_class','calling_method','called_class','called_method','called_method_args'])

    return df.reset_index(drop=True)

def follow_pattern_rtn(s):
    pattern = re.compile("^(.*?),\"(.*?)\",\"(.*?)", re.DOTALL)
    return True if pattern.match(s) else False


def split_line_rtn(s):
    pattern = re.compile("(.*?),\"(.*?)\",\"(.*?)\"", re.DOTALL)
    m = pattern.match(s)
    return [m.group(1), m.group(2), m.group(3)]


def read_rtn(rtn_path):
    with open(rtn_path) as file:
        lines = file.readlines()[1:]
        is_line = list(map(follow_pattern_rtn, lines))
        nlines = []
        for i, l in enumerate(lines):
            if is_line[i]:
                nlines.append(l)
            else:
                nlines[-1] = nlines[-1] + l
        nlines = list(map(split_line_rtn, nlines))
        df = pd.DataFrame(nlines, columns=['id', 'return_type' , 'return_value'])

    return df.reset_index(drop=True) 


def main():
    df = read_rtn('/home/emad/Desktop/defects4j-mutation/trace4j/projects/jacksoncore_2f_0/trace4j/logs/ret_log_com.fasterxml.jackson.core.json.TestJsonParser.csv')
    print(df)

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

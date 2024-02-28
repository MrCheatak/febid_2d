import glob
from pickle import load
import numpy as np
from pandas import read_csv
from tqdm import tqdm


def write_to_file(fname, names, *args, digits=5, append=False):
    n = len(args)
    if append:
        mode = 'a'
        header = '\t'.join(names) + '\n'
    else:
        mode = 'w'
        header = None
    with open(fname, mode=mode) as f:
        if header:
            header = '\t'.join(names) + '\n'
            f.write(header)
        text = ''
        for i in range(args[1].size):
            for j in range(n):
                val = np.round(args[j][i], digits)
                text += f'{val}'
                if j < n - 1:
                    text += '\t'
            text += '\n'
            f.write(text)
            text = ''


def extract_map(directory, *args, ext='txt'):
    """
    Extract parameters as columns from each Experiments Set file.

    :param ext: extension of files
    :param directory: directory with files
    :param args: names of parameters
    :return:
    """
    files = glob.glob(directory + r'\*.' + ext)
    data = [[] for i in args]
    for file in tqdm(files):
        with open(file, mode='rb') as f:
            obj = load(f)
            for i, arg in enumerate(args):
                data[i].append(obj.get_attr(arg))
    arrs = [np.array(d).reshape(-1) for d in data]
    return arrs

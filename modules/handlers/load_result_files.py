import json
import os

def collect_file_list(path):

    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              os.path.splitext(f)[1] == '.json']
    return result


def read_dict_data(files):

    data = []

    for file in files:
        with open(file, 'r') as json_file:
            data += [json.loads(json_file.read())]

    return data





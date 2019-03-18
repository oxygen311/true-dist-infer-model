import json

file_name = "graph_stats_300kb"


if __name__ == '__main__':
    data = json.loads(open(file_name, 'r').read())

    for (apcf, sp, way, n, cms, pms) in data:
        print(apcf, sp, way, n, cms, pms, sep="\n", end="\n\n")
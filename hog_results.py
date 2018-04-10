import sys
import json
import hog
from operator import itemgetter


if __name__ == '__main__':
    assert len(sys.argv) == 2
    assert sys.argv[1] in ['stl10', 'gtsrb', 'mnist', 'feret']

    check = {}
    for clf in ['KNN', 'SVM', 'RFC', 'LDA']:
        for cells_per_block in range(1,4):
            for pixels_per_cell in hog.hog_params[sys.argv[1]]['pixels_per_cell']:
                check[(clf, cells_per_block, pixels_per_cell[0])] = False

    filename = '%s.json' % sys.argv[1]
    filepath = hog.RESULTS_DIR / filename

    with filepath.open(mode='r') as f:
        data = [json.loads(line) for line in f.readlines()]

    data = sorted(data, key=itemgetter('score'))

    for trial in data:
        check[(trial['clf'], trial['cells_per_block'][0], trial['pixels_per_cell'][0])] = True

    for key in check:
        if not check[key]:
            print('Trial missing: %s, cells_per_block: %s, pixels_per_cell: %s' % key)

    print("----------BEST TRIALS----------")

    for trial in data[-10:]:
        print(trial)


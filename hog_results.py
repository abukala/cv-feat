import sys
import json
import hog


if __name__ == '__main__':
    assert len(sys.argv) == 2
    assert sys.argv[1] in ['stl10', 'gtsrb', 'mnist', 'feret']
    filename = '%s.json' % sys.argv[1]
    filepath = hog.RESULTS_DIR / filename

    with filepath.open(mode='r') as f:
        data = [json.loads(line) for line in f.readlines()]

    check = {}
    top_score = 0
    top_trial = None

    for clf in ['KNN', 'SVM', 'RFC', 'LDA']:
        for cells_per_block in range(1,4):
            for pixels_per_cell in hog.hog_params[sys.argv[1]]['pixels_per_cell']:
                check[(clf, cells_per_block, pixels_per_cell)] = False

    for trial in data:
        check[(trial['clf'], trial['cells_per_block'], trial['pixels_per_cell'])] = True
        if trial['score'] > top_score:
            top_score = trial['score']
            top_trial = trial

    for key in check:
        if not check[key]:
            print('Trial missing: %s, cells_per_block: %s, pixels_per_cell: %s' % key)

    print('Top trial: %s' % top_trial)
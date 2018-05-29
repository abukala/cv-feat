import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import databases
import os
import sys

sns.set_style('whitegrid')

DATASETS = ['gtsrb', 'stl10', 'feret', 'mnist']
CLASSIFIERS = ['KNN', 'LDA', 'SVM', 'RFC']
VISUALIZATIONS_PATH = os.path.join(os.path.dirname(__file__), 'visualizations')
DATABASE_PATH = databases.FINISHED_PATH


def visualize(noise_type, file_name=None):
    trials = pd.DataFrame(databases.select({
        'Noise_Type': noise_type
    }, fetch='all', database_path=DATABASE_PATH))
    trials_clean = pd.DataFrame(databases.select({
        'Noise_Type': 'none'
    }, fetch='all', database_path=DATABASE_PATH))
    noise_levels = trials['Noise_Level'].unique()
    for i, nl in enumerate(noise_levels):
        if nl == 'random':
            noise_levels = np.delete(noise_levels, i)
            break
    rows = []

    for dataset in DATASETS:
        for classifier in CLASSIFIERS:
            for feature in ['hog', 'none']:
                selection_clean = trials_clean[(trials_clean['Dataset'] == dataset) &
                                               (trials_clean['Classifier'] == classifier) &
                                               (trials_clean['Feature'] == feature)]
                for training_noise in ['yes', 'no']:
                    row = [dataset, classifier, 0, eval(selection_clean.iloc[0]['Score']), '%s_%s' % (feature, training_noise)]
                    rows.append(row)
            for noise_level in noise_levels:
                for feature in ['hog', 'none']:
                    for training_noise in ['yes', 'no']:
                        selection = trials[(trials['Dataset'] == dataset) &
                                           (trials['Train_Noise'] == training_noise) &
                                           (trials['Feature'] == feature) &
                                           (trials['Classifier'] == classifier) &
                                           (trials['Noise_Level'] == noise_level)]
                        assert len(selection) == 1

                        rows.append([dataset, classifier, eval(noise_level), eval(selection.iloc[0]['Score']), '%s_%s' % (feature, training_noise)])
    noise_levels = [eval(level) for level in noise_levels]
    if noise_type != 'lres':
        noise_levels = [round(level, 3) for level in noise_levels]
        noise_levels.insert(0, 0)
    df = pd.DataFrame(rows, columns=['dataset', 'classifier', 'noise level', 'accuracy', 'label'])
    grid = sns.FacetGrid(df, row='classifier', col='dataset', hue='label',  hue_kws={'ls': ['-', '-', '--', '--']}, legend_out=True)
    grid.set(ylim=(0.0, 1.0), xticks=range(len(noise_levels)))
    grid.set_xticklabels(noise_levels, rotation=90)
    grid.map(plt.plot, 'accuracy')
    grid.fig.legend(loc='lower center', ncol=4, labels=['hog with train noise', 'hog without train noise', 'vectorized with train noise', 'vectorized without train noise'],)
    grid.fig.subplots_adjust(bottom=0.10)
    grid.set_xlabels('noise level')
    grid.set_ylabels('accuracy')

    if file_name is None:
        plt.show()
    else:
        if not os.path.exists(VISUALIZATIONS_PATH):
            os.mkdir(VISUALIZATIONS_PATH)

        grid.savefig(os.path.join(VISUALIZATIONS_PATH, file_name))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        DATABASE_PATH = pathlib.Path(sys.argv[1])
    for feature in ['none', 'hog']:
        for noise_type in ['gauss', 'sp', 'quantization', 'blur', 'occlusion']:
            visualize(noise_type, '%s.pdf' % noise_type)
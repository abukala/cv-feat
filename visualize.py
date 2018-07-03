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


def visualize(noise_type, type, file_name):
    assert type in ['denoise', 'hogvec']
    trials = pd.DataFrame(databases.select({
        'Noise_Type': noise_type,
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
            if type == 'denoise':
                selection_clean = trials_clean[(trials_clean['Dataset'] == dataset) &
                                               (trials_clean['Classifier'] == classifier) &
                                               (trials_clean['Feature'] == 'hog')]
                for label in ['n', 'tn', 'dn']:
                    rows.append([dataset.upper(), classifier, 0, eval(selection_clean.iloc[0]['Score']), label])
            elif type == 'hogvec':
                for feature in ['hog', 'none']:
                    selection_clean = trials_clean[(trials_clean['Dataset'] == dataset) &
                                                   (trials_clean['Classifier'] == classifier) &
                                                   (trials_clean['Feature'] == feature)]
                    for training_noise in ['yes', 'no']:
                        rows.append([dataset.upper(), classifier, 0, eval(selection_clean.iloc[0]['Score']), '%s_%s' % (feature, training_noise)])
            for noise_level in noise_levels:
                if type == 'denoise':
                    for training_noise in ['yes', 'no']:
                        selection = trials[(trials['Dataset'] == dataset) &
                                           (trials['Train_Noise'] == training_noise) &
                                           (trials['Feature'] == 'hog') &
                                           (trials['Classifier'] == classifier) &
                                           (trials['Noise_Level'] == noise_level) &
                                           (trials['Denoise_Params'].isnull())]
                        assert len(selection) == 1, (dataset, training_noise, classifier, noise_type, noise_level, len(selection))
                        if training_noise == 'yes':
                            label = 'tn'
                        else:
                            label = 'n'
                        rows.append([dataset.upper(), classifier, eval(noise_level), eval(selection.iloc[0]['Score']), label])
                    selection = trials[(trials['Dataset'] == dataset) &
                                       (trials['Train_Noise'] == 'no') &
                                       (trials['Feature'] == 'hog') &
                                       (trials['Classifier'] == classifier) &
                                       (trials['Noise_Level'] == noise_level) &
                                       (trials['Denoise_Params'].notnull())]
                    assert len(selection) == 1, (dataset, training_noise, classifier, noise_type, noise_level, len(selection))
                    rows.append([dataset.upper(), classifier, eval(noise_level), eval(selection.iloc[0]['Score']), 'dn'])
                elif type == 'hogvec':
                    for feature in ['hog', 'none']:
                        for train_noise in ['yes', 'no']:
                            selection = trials[(trials['Dataset'] == dataset) &
                                               (trials['Train_Noise'] == train_noise) &
                                               (trials['Feature'] == feature) &
                                               (trials['Classifier'] == classifier) &
                                               (trials['Noise_Level'] == noise_level) &
                                               (trials['Denoise_Params'].isnull())]
                            assert len(selection) == 1, (dataset, classifier, noise_type, noise_level, len(selection))

                            rows.append([dataset.upper(), classifier, eval(noise_level), eval(selection.iloc[0]['Score']), '%s_%s' % (feature, train_noise)])
    noise_levels = [eval(level) for level in noise_levels]
    if noise_type != 'lres':
        noise_levels = [round(level, 3) for level in noise_levels]
        noise_levels.insert(0, 0)

    df = pd.DataFrame(rows, columns=['dataset', 'classifier', 'noise level', 'accuracy', 'label'])
    if type == 'denoise':
        hue_kws = {}
        hue_order = ['n', 'tn', 'dn']
        labels = ['distorted images', 'noise applied on training set', 'applied denoising']
        ncol = 3
    if type == 'hogvec':
        hue_kws = {'ls': ['-', '-', '--', '--']}
        hue_order = ['hog_yes', 'hog_no', 'none_yes', 'none_no']
        labels = ['HoG descriptors with training noise', 'HoG descriptors without training noise', 'vectorized with training noise', 'vectorized without training noise' ]
        ncol = 2
    grid = sns.FacetGrid(df, row='classifier', col='dataset', hue='label', hue_kws=hue_kws, hue_order=hue_order, legend_out=True, size=2.5, gridspec_kws={"wspace":0.1})
    grid.set(ylim=(0.0, 1.0), xticks=range(len(noise_levels)))
    grid.set_xticklabels(noise_levels, rotation=90)
    grid.map(plt.plot, 'accuracy')
    grid.set_titles("{row_name} | {col_name}")
    grid.fig.legend(loc='lower center', ncol=ncol, labels=labels)
    grid.fig.subplots_adjust(bottom=0.15)
    grid.set_xlabels('noise level')
    grid.set_ylabels('accuracy')

    if not os.path.exists(VISUALIZATIONS_PATH):
        os.mkdir(VISUALIZATIONS_PATH)

    grid.savefig(os.path.join(VISUALIZATIONS_PATH, file_name))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        DATABASE_PATH = pathlib.Path(sys.argv[1])
    for noise_type in ['gauss', 'sp', 'quantization', 'blur', 'occlusion']:
        visualize(noise_type, 'hogvec', '%s.pdf' % noise_type)
    for noise_type in ['gauss', 'sp', 'quantization']:
        visualize(noise_type, 'denoise', '%s_denoise.pdf' % noise_type)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import databases
import os


DATASETS = ['gtsrb', 'stl10', 'cifar10', 'mnist']
CLASSIFIERS = ['KNN', 'LDA', 'SVM', 'RFC']
VISUALIZATIONS_PATH = os.path.join(os.path.dirname(__file__), 'visualizations')


def visualize(feature, noise_type, file_name=None):
    trials = pd.DataFrame(databases.select({
        'Feature': feature,
        'Noise_Type': noise_type
    }, fetch='all', database_path=databases.FINISHED_PATH))

    trials.loc[trials['Noise_Level'] == 'none', 'Noise_Level'] = 0
    noise_levels = trials['Noise_Level'].unique()
    rows = []

    for dataset in DATASETS:
        for classifier in CLASSIFIERS:
            for noise_level in noise_levels:
                selection = trials[(trials['Dataset'] == dataset) & (trials['Classifier'] == classifier) & (trials['Noise_Level'] == noise_level)]

                if noise_level == 0:
                    assert len(selection) == 1

                    for training_noise in ['yes', 'no']:
                        rows.append([dataset, classifier, noise_level, eval(selection.iloc[0]['Score']), training_noise])
                else:
                    for training_noise in ['yes', 'no']:
                        subselection = selection[selection['Train_Noise'] == training_noise]

                        assert len(subselection) == 1

                        rows.append([dataset, classifier, eval(noise_level), eval(subselection.iloc[0]['Score']), training_noise])
    noise_levels = [eval(level) for level in noise_levels]
    if noise_type is not 'lres':
        noise_levels = [round(level, 2) for level in noise_levels]
    df = pd.DataFrame(rows, columns=['dataset', 'classifier', 'noise level', 'accuracy', 'training noise'])
    grid = sns.FacetGrid(df, row='classifier', col='dataset', hue='training noise', legend_out=True)
    grid.set(ylim=(0.0, 1.0), xticks=range(len(noise_levels)))
    grid.set_xticklabels(noise_levels, rotation=90)
    grid.map(plt.plot, 'accuracy')
    grid.set_xlabels('noise level')
    grid.set_ylabels('accuracy')

    if file_name is None:
        plt.show()
    else:
        if not os.path.exists(VISUALIZATIONS_PATH):
            os.mkdir(VISUALIZATIONS_PATH)

        grid.savefig(os.path.join(VISUALIZATIONS_PATH, file_name))


if __name__ == '__main__':
    for feature in ['none', 'hog']:
        for noise_type in ['gauss', 'sp', 'quantization', 'lres', 'occlusion']:
            visualize(feature, noise_type, '%s_%s.pdf' % (feature, noise_type))
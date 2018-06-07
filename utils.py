from databases import RESULTS_PATH
RESULTS_PATH = RESULTS_PATH / 'baseline'


def get_params():
    params = {
        'gtsrb': {
            'hog': {
                'pixels_per_cell': (4, 4),
                'cells_per_block': (2, 2)
            },
            'clf': {
                'hog': {
                    'RFC': {
                        'n_estimators': 100
                    },
                    'SVM': {
                        'C': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 25
                    }
                },
                'none': {
                    'RFC': {
                        'n_estimators': 100
                    },
                    'SVM': {
                        'C': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 1
                    }
                }
            }
        },
        'stl10': {
            'hog': {
                'pixels_per_cell': (16, 16),
                'cells_per_block': (3, 3)
            },
            'clf': {
                'hog': {
                    'SVM': {
                        'C': 100
                    },
                    'RFC': {
                        'n_estimators': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 10
                    }
                },
                'none': {
                    'SVM': {
                        'C': 10
                    },
                    'RFC': {
                        'n_estimators': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 1
                    }
                }
            }
        },
        'mnist': {
            'hog': {
                'pixels_per_cell': (4, 4),
                'cells_per_block': (1, 1)
            },
            'clf': {
                'hog': {
                    'RFC': {
                        'n_estimators': 100
                    },
                    'SVM': {
                        'C': 10
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 5
                    }
                },
                'none': {
                    'RFC': {
                        'n_estimators': 100
                    },
                    'SVM': {
                        'C': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 1
                    }
                }
            }
        },
        'feret': {
            'hog': {
                'pixels_per_cell': (32, 32),
                'cells_per_block': (1, 1)
            },
            'clf': {
                'hog': {
                    'RFC': {
                        'n_estimators': 100
                    },
                    'SVM': {
                        'C': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 1
                    }
                },
                'none': {
                    'RFC': {
                        'n_estimators': 100
                    },
                    'SVM': {
                        'C': 100
                    },
                    'LDA': {
                        'n_components': 1
                    },
                    'KNN': {
                        'n_neighbors': 1
                    }
                }
            }
        },
    }
    for dataset in ['gtsrb', 'stl10', 'feret', 'mnist']:
        params[dataset]['denoise'] = {}
        for noise_type in ['gauss', 'sp', 'quantization']:
            params[dataset]['denoise'][noise_type] = {}
            filename = '%s_%s.json' % (dataset, noise_type)
            fp = RESULTS_PATH / filename
            with fp.open() as file:
                data = [eval(line) for line in file.readlines()]
            for result in data:
                scores = {x: result[x][1] for x in ['bm3d', 'bilateral', 'median']}
                best_method = max(scores, key=scores.get)
                if result['noise_level'] != 'random':
                    noise_level = round(result['noise_level'], 3)
                else:
                    noise_level = result['noise_level']
                params[dataset]['denoise'][noise_type][noise_level] = (best_method, result[best_method][0])
        filename = '%s_random.json' % dataset
        fp = RESULTS_PATH / filename
        with fp.open() as file:
            data = [eval(line) for line in file.readlines()]
        result = data[0]
        scores = {x: result[x][1] for x in ['bm3d', 'bilateral', 'median']}
        best_method = max(scores, key=scores.get)
        params[dataset]['denoise']['random'] = (best_method, result[best_method][0])

    return params

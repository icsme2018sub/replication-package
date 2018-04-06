import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    mean_absolute_error, make_scorer
from sklearn.utils import shuffle


def plot_feature_importance(metrics, name):
    """
    Draws a bar plot of the metrics passed as parameter;
    It requires a plotly account to be locally installed on the machine
    :param metrics: the metrics to plot
    :param name: the name of the metrics
    """
    values, names = zip(*metrics)
    names = [x.replace('_prod', ' prod.') for x in names]
    names = [x.replace('_test', ' test') for x in names]
    names = [x.replace('csm_', '') for x in names]
    names = [x.replace('_', ' ') for x in names]
    names = [x.replace('is', ' ') for x in names]

    data = [go.Bar(
        x=values,
        y=names,
        orientation='h',
    )]

    layout = go.Layout(
        xaxis=dict(
            titlefont=dict(
              size=14
            ),
            tickfont=dict(
                size=14
            ),
        ),
        yaxis=dict(
            titlefont=dict(
                size=14
            ),
            tickfont=dict(
                size=14
            ),
        ),
        autosize=True,
        margin=go.Margin(
            l=170,
            r=50,
            b=50,
            t=20,
            pad=10
        ),
        legend=dict(
            font=dict(
                size=6
            ),
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.image.save_as(fig, filename=name+'.pdf')


def classify(coverage=True):
    """
    Trains and evaluates the classify;
    At first, it performs a grid search over the possible parameters on the half of the available dataset;
    Then, gets the best parameters, trains and evaluate the model of the remaining half part of the dataset.
    :param coverage: if True, line coverage is considered as a factor for the training process;
    """
    good_tests = pd.read_csv('good_tests.csv')
    bad_tests = pd.read_csv('bad_tests.csv')
    start_index = 8 if coverage else 9
    metrics = good_tests.columns[start_index:-10]
    l = len(metrics)
    good_tests['y'] = good_tests.apply(lambda x: 1, axis=1)
    bad_tests['y'] = bad_tests.apply(lambda x: 0, axis=1)

    frame = shuffle(pd.concat([good_tests, bad_tests]))

    n_tuning = 10
    n_validation = 100

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    maes = []

    features = [[] for _ in range(len(metrics))]

    print("* Splitting dataset for parameter tuning and validation set\n")
    X_tuning, X_valid, y_tuning, y_valid = train_test_split(frame[metrics], frame['y'], stratify=frame['y'],
                                                            train_size=0.5, test_size=0.5, random_state=0)

    print("* Size tuning set = {}:".format(X_tuning.shape[0]))
    print("- Good tests = {}".format(y_tuning[y_tuning == 1].shape[0]))
    print("- Bad tests = {}".format(y_tuning[y_tuning == 0].shape[0]))
    print("\n* Size training set= {}".format(y_valid.shape[0]))
    print("- Good tests = {}".format(y_valid[y_valid == 1].shape[0]))
    print("- Bad tests = {}".format(y_valid[y_valid == 0].shape[0]))

    best_score = 0
    best_parameter = {}

    print("\n* Starting parameter tuning:")
    for i in range(n_tuning):
        print("* Run #{}".format(i+1))

        k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        param_grid = {'n_estimators': [3*x for x in range(1, 11)],
                      'max_features': [int((l/10)*x) for x in range(1, 11)],
                      'max_depth': [5*x for x in range(1, 11)],
                      'min_samples_leaf': [2*x for x in range(1, 11)]}

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=k_fold, n_jobs=-1, verbose=0,
                                   return_train_score=True)
        grid_search.fit(X_tuning, y_tuning)
        score = grid_search.best_score_
        if score > best_score:
            best_score = score
            best_parameter = grid_search.best_params_
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('grid_search/run_{}.csv'.format(i), index=False)
        print("* Best parameters: {}".format(grid_search.best_params_))

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score),
               'roc_auc_scorer': make_scorer(roc_auc_score),
               'mean_absolute_error': make_scorer(mean_absolute_error)}

    print("\n* Starting cross validation after parameter tuning:")
    print("* Using the following parameters:")
    print(best_parameter)
    for i in range(n_validation):
        print("* Run #{}".format(i + 1))
        model = RandomForestClassifier(**best_parameter)
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        results = cross_validate(estimator=model, X=X_valid, y=y_valid, cv=k_fold, scoring=scoring)
        model.fit(X_valid, y_valid)
        for j, elem in enumerate(model.feature_importances_):
            features[j].append(elem)

            accuracies.append(results.get('test_accuracy').mean())
            precisions.append(results.get('test_precision').mean())
            recalls.append(results.get('test_recall').mean())
            f1_scores.append(results.get('test_f1_score').mean())
            roc_aucs.append(results.get('test_roc_auc_scorer').mean())
            maes.append(results.get('test_mean_absolute_error').mean())

    mean = lambda x: sum(x) / len(x)

    print("\nOverall performances")
    print("\nAccuracy: {:.3f}".format(mean(accuracies)))
    print("Precision: {:.3f}".format(mean(precisions)))
    print("Recall: {:.3f}".format(mean(recalls)))
    print("F1 Score: {:.3f}".format(mean(f1_scores)))
    print("Roc AUC: {:.3f}".format(mean(roc_aucs)))
    print("Mean Absolute Error: {:.3f}".format(mean(maes)))

    runs = list(range(n_validation))
    estimation = pd.DataFrame(features, columns=runs, index=metrics)
    name = 'estimation_with.csv' if coverage else 'estimation_no_cov.csv'
    estimation.to_csv(name)

    mean = lambda x: sum(x) / len(x)
    features_importance = [mean(x) for x in features]

    s = sorted(zip(map(lambda x: round(x, 3), features_importance), metrics), reverse=True)
    print(s[0:20])
    name = 'with_cov' if coverage else 'no_cov'
    plot_feature_importance(s[0:20], name)


if __name__ == '__main__':
    classify()
    classify(coverage=False)



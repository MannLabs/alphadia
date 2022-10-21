"""Percolate results."""

import logging

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.decomposition
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.pipeline


import alphatims.utils


class Percolator:

    def __init__(
        self,
        fdr=0.01,
        train_fdr_level_pre_calibration=0.1,
        train_fdr_level_post_calibration=0.33,
        n_neighbors=4,
        test_size=0.5,
        random_state=0,
    ):
        self.fdr = fdr
        self.train_fdr_level_pre_calibration = train_fdr_level_pre_calibration
        self.train_fdr_level_post_calibration = train_fdr_level_post_calibration
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state

    def set_annotation(self, annotation):
        self.annotation = annotation

    def percolate(self):
        logging.info("Percolating PSMs")
        val_names = [
            "counts",
            "frequency_counts",
            "ppm_diff",
            "im_diff",
            "charge",
            "total_peaks",
            "nAA",
            "b_hit_counts",
            "y_hit_counts",
            "b_mean_ppm",
            "y_mean_ppm",
            "relative_found_b_int",
            "relative_missed_b_int",
            "relative_found_y_int",
            "relative_missed_y_int",
            "relative_found_int",
            "relative_missed_int",
            "pearsons",
            "pearsons_log",
            "candidates",
        ]
        logging.info("Calculating quick log odds")
        score_df = self.annotation.copy()
        log_odds = calculate_log_odds_product(
            score_df,
            val_names,
        )
        # log_odds = score_df["frequency_counts"].values
        score_df["log_odds"] = log_odds
        # score_df = alphadia.prefilter.train_and_score(
        #     score_df,
        #     val_names,
        #     ini_score="log_odds",
        #     train_fdr_level=train_fdr_level_pre_calibration,
        # ).reset_index(drop=True)
        score_df = get_q_values(score_df, "log_odds", 'decoy', drop=True)
        score_df_above_fdr = score_df[
            (score_df.q_value < self.fdr) & (score_df.target)
        ].reset_index(drop=True)
        logging.info(
            f"Found {len(score_df_above_fdr)} targets for calibration"
        )
        score_df_above_fdr["im_pred"] = score_df_above_fdr.mobility_pred
        score_df_above_fdr["im_values"] = score_df_above_fdr.mobility_values
        self.predictors = {}
        for dimension in ["rt", "im"]:
            X = score_df_above_fdr[f"{dimension}_pred"].values.reshape(-1, 1)
            y = score_df_above_fdr[f"{dimension}_values"].values
            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = sklearn.model_selection.train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            self.predictors[dimension] = sklearn.neighbors.KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                # weights="distance",
                n_jobs=alphatims.utils.set_threads(alphatims.utils.MAX_THREADS)
            )
            self.predictors[dimension].fit(X_train, y_train)
            score_df_above_fdr[f"{dimension}_calibrated"] = self.predictors[dimension].predict(
                score_df_above_fdr[f"{dimension}_pred"].values.reshape(-1, 1)
            )
            score_df_above_fdr[f"{dimension}_diff"] = score_df_above_fdr[f"{dimension}_values"] - score_df_above_fdr[f"{dimension}_calibrated"]
        score_df["rt_calibrated"] = self.predictors["rt"].predict(
            score_df.rt_pred.values.reshape(-1, 1)
        )
        score_df["im_calibrated"] = self.predictors["im"].predict(
            score_df.mobility_pred.values.reshape(-1, 1)
        )
        ppm_mean = np.mean(score_df_above_fdr.ppm_diff.values)
        score_df["mz_calibrated"] = score_df.precursor_mz * (
            1 - ppm_mean * 10**-6
        )

        score_df["ppm_diff_calibrated"] = (score_df.mz_calibrated - score_df.mz_values) / score_df.mz_calibrated * 10**6
        score_df["rt_diff_calibrated"] = score_df.rt_calibrated - score_df.rt_values
        score_df["im_diff_calibrated"] = score_df.im_calibrated - score_df.mobility_values
        # self.score_df = score_df.reset_index(drop=True)
        self.score_df = train_and_score(
            # score_df[np.abs(score_df.rt_diff_calibrated) < 250].reset_index(drop=True),
            score_df,
            [
                "counts",
                "frequency_counts",
                "ppm_diff_calibrated",
                "im_diff_calibrated",
                "rt_diff_calibrated",
                "charge",
                "total_peaks",
                "nAA",
                "b_hit_counts",
                "y_hit_counts",
                "b_mean_ppm",
                "y_mean_ppm",
                "relative_found_b_int",
                "relative_missed_b_int",
                "relative_found_y_int",
                "relative_missed_y_int",
                "relative_found_int",
                "relative_missed_int",
                "pearsons",
                "pearsons_log",
                "candidates",
                # "log_odds",
            ],
            ini_score="log_odds",
            train_fdr_level=self.train_fdr_level_post_calibration,
        ).reset_index(drop=True)

        self.score_df["target_type"] = np.array([-1, 0])[
            self.score_df.target.astype(np.int)
        ]
        self.score_df["target_type"][
            (self.score_df.q_value < self.fdr) & (self.score_df.target)
        ] = 1


@alphatims.utils.njit(nogil=True)
def fdr_to_q_values(fdr_values):
    q_values = np.zeros_like(fdr_values)
    min_q_value = np.max(fdr_values)
    for i in range(len(fdr_values) - 1, -1, -1):
        fdr = fdr_values[i]
        if fdr < min_q_value:
            min_q_value = fdr
        q_values[i] = min_q_value
    return q_values


def get_q_values(_df, score_column, decoy_column, drop=False):
    _df = _df.reset_index(drop=drop)
    _df = _df.sort_values([score_column, score_column], ascending=False)
    target_values = 1-_df['decoy'].values
    decoy_cumsum = np.cumsum(_df['decoy'].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum/target_cumsum
    _df['q_value'] = fdr_to_q_values(fdr_values)
    return _df


def calculate_odds(
    df,
    column_name,
    *,
    target_name="target",
    smooth=1,
    plot=False
):
    negatives, positives = np.bincount(df.target.values)
    if negatives > positives:
        raise ValueError(
            f"Found more decoys ({negatives}) than targets ({positives})"
        )
        tp_count = 1000
    else:
        tp_count = positives - negatives
    n = int(tp_count * smooth)
    order = np.argsort(df[column_name].values)
    forward = np.cumsum(df[target_name].values[order])
    odds = np.zeros_like(forward, dtype=np.float)
    odds[n:-n] = forward[2*n:] - forward[:-2*n]
    odds[:n] = forward[n:2*n]
    odds[-n:] = forward[-1] - forward[-2*n:-n]
    odds[n:-n] /= 2*n
    odds[:n] /= np.arange(n, 2*n)
    odds[-n:] /= np.arange(n, 2*n)[::-1]
    odds /= (1 - odds)
    odds = odds[np.argsort(order)]
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(df[column_name], odds, marker=".")
    return odds


def calculate_log_odds_product(
    df_,
    val_names
):
    df = df_[val_names]
    df = sklearn.preprocessing.StandardScaler().fit_transform(df)
    pca = sklearn.decomposition.PCA(n_components=df.shape[1])
    pca.fit(df)
    df = pd.DataFrame(pca.transform(df))
    df["target"] = df_.target
    negative, positive = np.bincount(df.target)
    log_odds = np.zeros(len(df))
    for val_name in range(df.shape[1] - 1):
        odds = calculate_odds(df, val_name, smooth=1)
        log_odds += np.log2(odds) * pca.explained_variance_[val_name]
    return log_odds
    # new_df = analysis1.score_df[["decoy", "target"]]
    # new_df['odds'] = log_odds
    # new_df = alphadia.library.get_q_values(new_df, "odds", 'decoy', drop=True)
    # new_df.reset_index(drop=True, inplace=True)


def train_and_score(
    scores_df,
    features,
    train_fdr_level: float = 0.1,
    ini_score: str = "count",
    min_train: int = 1000,
    test_size: float = 0.8,
    max_depth: list = [5, 25, 50],
    max_leaf_nodes: list = [150, 200, 250],
    n_jobs: int = -1,
    scoring: str = 'accuracy',
    plot: bool = False,
    random_state: int = 42,
):
    df = scores_df.copy()
    cv = train_RF(
        df,
        features,
        train_fdr_level=train_fdr_level,
        ini_score=ini_score,
        min_train=min_train,
        test_size=test_size,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        n_jobs=n_jobs,
        scoring=scoring,
        plot=plot,
        random_state=random_state,
    )
    df['score'] = cv.predict_proba(df[features])[:, 1]
    return get_q_values(df, "score", 'decoy', drop=True)


def train_RF(
    df: pd.DataFrame,
    features: list,
    train_fdr_level:  float = 0.1,
    ini_score: str = None,
    min_train: int = 1000,
    test_size: float = 0.8,
    max_depth: list = [5, 25, 50],
    max_leaf_nodes: list = [150, 200, 250],
    n_jobs: int = -1,
    scoring: str = 'accuracy',
    plot: bool = False,
    random_state: int = 42,
):
    # Setup ML pipeline
    scaler = sklearn.preprocessing.StandardScaler()
    rfc = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
    ## Initiate scaling + classification pipeline
    pipeline = sklearn.pipeline.Pipeline([('scaler', scaler), ('clf', rfc)])
    parameters = {
        'clf__max_depth': (max_depth),
        'clf__max_leaf_nodes': (max_leaf_nodes)
    }
    ## Setup grid search framework for parameter selection and internal cross validation
    cv = sklearn.model_selection.GridSearchCV(
        pipeline,
        param_grid=parameters,
        cv=5,
        scoring=scoring,
        verbose=0,
        return_train_score=True,
        n_jobs=n_jobs
    )
    # Prepare target and decoy df
    dfD = df[df.decoy.values]
    # Select high scoring targets (<= train_fdr_level)
    # df_prescore = filter_score(df)
    # df_prescore = filter_precursor(df_prescore)
    # scored = cut_fdr(df_prescore, fdr_level = train_fdr_level, plot=False)[1]
    # highT = scored[scored.decoy==False]
    # dfT_high = dfT[dfT['query_idx'].isin(highT.query_idx)]
    # dfT_high = dfT_high[dfT_high['db_idx'].isin(highT.db_idx)]
    if ini_score is None:
        selection = None
        best_hit_count = 0
        best_feature = ""
        for feature in features:
            new_df = get_q_values(df, feature, 'decoy')
            hits = (
                new_df['q_value'] <= train_fdr_level
            ) & (
                new_df['decoy'] == 0
            )
            hit_count = np.sum(hits)
            if hit_count > best_hit_count:
                best_hit_count = hit_count
                selection = hits
                best_feature = feature
        logging.info(f'Using optimal "{best_feature}" as initial_feature')
        dfT_high = df[selection]
    else:
        logging.info(f'Using selected "{ini_score}" as initial_feature')
        new_df = get_q_values(df, ini_score, 'decoy')
        dfT_high = df[
            (new_df['q_value'] <= train_fdr_level) & (new_df['decoy'] == 0)
        ]

    # Determine the number of psms for semi-supervised learning
    n_train = int(dfT_high.shape[0])
    if dfD.shape[0] < n_train:
        n_train = int(dfD.shape[0])
        logging.info(
            "The total number of available decoys is lower than "
            "the initial set of high scoring targets."
        )
    if n_train < min_train:
        raise ValueError(
            "There are fewer high scoring targets or decoys than "
            "required by 'min_train'."
        )

    # Subset the targets and decoys datasets to result in a balanced dataset
    df_training = dfT_high.append(
        dfD.sample(n=n_train, random_state=random_state)
    )
    # df_training = dfT_high.append(dfD)

    # Select training and test sets
    X = df_training[features]
    y = df_training['target'].astype(int)
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = sklearn.model_selection.train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values
    )

    # Train the classifier on the training set via 5-fold cross-validation and subsequently test on the test set
    logging.info(
        'Training & cross-validation on {} targets and {} decoys'.format(
            # np.sum(y_train), X_train.shape[0] - np.sum(y_train)
            *np.bincount(y_train)[::-1]
        )
    )
    cv.fit(X_train, y_train)

    logging.info(
        'The best parameters selected by 5-fold cross-validation were {}'.format(
            cv.best_params_
        )
    )
    logging.info(
        'The train {} was {}'.format(scoring, cv.score(X_train, y_train))
    )
    logging.info(
        'Testing on {} targets and {} decoys'.format(
            np.sum(y_test),
            X_test.shape[0] - np.sum(y_test)
        )
    )
    logging.info(
        'The test {} was {}'.format(scoring, cv.score(X_test, y_test))
    )

    feature_importances = cv.best_estimator_.named_steps['clf'].feature_importances_
    indices = np.argsort(feature_importances)[::-1][:40]

    top_features = X.columns[indices][:40]
    top_score = feature_importances[indices][:40]

    feature_dict = dict(zip(top_features, top_score))
    logging.info(f"Top features {feature_dict}")

    # Inspect feature importances
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        g = sns.barplot(
            y=X.columns[indices][:40],
            x=feature_importances[indices][:40],
            orient='h',
            palette='RdBu'
        )
        g.set_xlabel("Relative importance", fontsize=12)
        g.set_ylabel("Features", fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title("Feature importance")
        plt.show()

    return cv

import os
import typing
import pickle
import logging
from typing import Literal

import pandas as pd
import numpy as np
import xxhash
import numba as nb


logger = logging.getLogger()
if not 'progress' in dir(logger):
    from alphadia.extraction import processlogger
    processlogger.init_logging()

import alphadia
from alphadia.extraction import calibration
import sklearn
import matplotlib.pyplot as plt

class BaseManager():

    def __init__(
            self,
            path : typing.Union[None, str] = None,
            load_from_file : bool = True
        ):

        """Base class for all managers which handle parts of the workflow.
        
        Parameters
        ----------

        path : str, optional
            Path to the manager pickle on disk.

        load_from_file : bool, optional
            If True, the manager will be loaded from file if it exists.
        """

        self._path = path
        self.is_loaded_from_file = False
        self.is_fitted = False

        self._version = alphadia.__version__

        if load_from_file:
            self.load()

    @property
    def path(self):
        """Path to the manager pickle on disk.
        """
        return self._path

    @property
    def is_loaded_from_file(self):
        """Check if the calibration manager was loaded from file.
        """
        return self._is_loaded_from_file
    
    @is_loaded_from_file.setter
    def is_loaded_from_file(self, value):
        self._is_loaded_from_file = value
    
    @property
    def is_fitted(self):
        """Check if all estimators in all calibration groups are fitted.
        """
        return self._is_fitted
    
    @is_fitted.setter
    def is_fitted(self, value):
        self._is_fitted = value
    
    def save(self):
        """Save the state to pickle file.
        """
        if self.path is not None:
            try:
                with open(self.path, 'wb') as f:
                    pickle.dump(self, f)
            except:
                logging.error(f'Failed to save {self.__class__.__name__} to {self.path}')

    def load(self):

        """Load the state from pickle file.
        """
        if self.path is not None:
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'rb') as f:
                        loaded_state = pickle.load(f)

                        if loaded_state._version == self._version:
                            self.__dict__.update(loaded_state.__dict__)
                            self.is_loaded_from_file = True
                        else:
                            logging.warning(f'Version mismatch while loading {self.__class__}: {loaded_state._version} != {self._version}. Will not load.')
                except:
                    logging.error(f'Failed to load {self.__class__.__name__} from {self.path}')
                else:
                    logging.info(f'Loaded {self.__class__.__name__} from {self.path}')
            else:
                logging.warning(f'{self.__class__.__name__} not found at: {self.path}')

    
    def fit(self):
        """Fit the workflow property of the manager.
        """
        raise NotImplementedError(f'fit() not implemented for {self.__class__.__name__}')
    
    def predict(self):
        """Return the predictions of the workflow property of the manager.
        """
        raise NotImplementedError(f'predict() not implemented for {self.__class__.__name__}')
    
    def fit_predict(self):
        """Fit and return the predictions of the workflow property of the manager.
        """
        raise NotImplementedError(f'fit_predict() not implemented for {self.__class__.__name__}')
    
class CalibrationManager(BaseManager):

    def __init__(
            self,
            config : typing.Union[None, dict] = None,
            path : typing.Union[None, str] = None,
            load_from_file : bool = True
        ):

        """Contains, updates and applies all calibrations for a single run.

        Calibrations are grouped into calibration groups. Each calibration group is applied to a single data structure (precursor dataframe, fragment fataframe, etc.). Each calibration group contains multiple estimators which each calibrate a single property (mz, rt, etc.). Each estimator is a `Calibration` object which contains the estimator function.
        
        Parameters
        ----------

        config : typing.Union[None, dict], default=None
            Calibration config dict. If None, the default config is used.

        path : str, default=None
            Path where the current parameter set is saved to and loaded from.

        load_from_file : bool, default=True
            If True, the manager will be loaded from file if it exists.
        
        """
        logging.info('========= Initializing Calibration Manager =========')
        super().__init__(path=path, load_from_file=load_from_file)

        if not self.is_loaded_from_file:
            self.estimator_groups = []
            self.load_config(config)
        
        logging.info('====================================================')


    @property
    def estimator_groups(self):
        """List of calibration groups.
        """
        return self._estimator_groups
    
    @estimator_groups.setter
    def estimator_groups(self, value):
        self._estimator_groups = value


    def load_config(self, config : dict):
        """Load calibration config from config Dict.

        each calibration config is a list of calibration groups which consist of multiple estimators.
        For each estimator the `model` and `model_args` are used to request a model from the calibration_model_provider and to initialize it.
        The estimator is then initialized with the `Calibration` class and added to the group.

        Parameters
        ----------

        config : dict
            Calibration config dict

        Example
        -------

        Create a calibration manager with a single group and a single estimator:

        .. code-block:: python

            calibration_manager = calibration.CalibrationManager()
            calibration_manager.load_config([{
                'name': 'mz_calibration',
                'estimators': [
                    {
                        'name': 'mz',
                        'model': 'LOESSRegression',
                        'model_args': {
                            'n_kernels': 2
                        },
                        'input_columns': ['mz_library'],
                        'target_columns': ['mz_observed'],
                        'output_columns': ['mz_calibrated'],
                        'transform_deviation': 1e6
                    },
                    
                ]
            }])
        
        """
        
        logging.info('loading calibration config')
        logging.info(f'found {len(config)} calibration groups')
        for group in config:
            logging.info(f'Calibration group :{group["name"]}, found {len(group["estimators"])} estimator(s)')
            for estimator in group['estimators']:
                try:
                    template = calibration.calibration_model_provider.get_model(estimator['model'])
                    model_args = estimator['model_args'] if 'model_args' in estimator else {}
                    estimator['function'] = template(**model_args)
                except Exception as e:
                    logging.error(f'Could not load estimator {estimator["name"]}: {e}')

            group_copy = {'name': group['name']} 
            group_copy['estimators'] = [calibration.Calibration(**x) for x in group['estimators']]
            self.estimator_groups.append(group_copy)

    def get_group_names(self):
        """Get the names of all calibration groups.

        Returns
        -------
        list of str
            List of calibration group names
        """

        return [x['name'] for x in self.estimator_groups]

    def get_group(self, group_name : str):
        """Get the calibration group by name.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        Returns
        -------
        dict
            Calibration group dict with `name` and `estimators` keys\
        
        """
        for group in self.estimator_groups:
            if group['name'] == group_name:
                return group

        logging.error(f'could not get_group: {group_name}')
        return None
    
    def get_estimator_names(self, group_name : str):
        """Get the names of all estimators in a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        Returns
        -------
        list of str
            List of estimator names
        """

        group = self.get_group(group_name)
        if group is not None:
            return [x.name for x in group['estimators']]
        logging.error(f'could not get_estimator_names: {group_name}')
        return None

    def get_estimator(self, group_name : str, estimator_name : str):

        """Get an estimator from a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        estimator_name : str
            Name of the estimator

        Returns
        -------
        Calibration
            The estimator object

        """
        group = self.get_group(group_name)
        if group is not None:
            for estimator in group['estimators']:
                if estimator.name == estimator_name:
                    return estimator
        logging.error(f'could not get_estimator: {group_name}, {estimator_name}')
        return None

    def fit(
        self, 
        df : pd.DataFrame, 
        group_name : str, 
        *args,
        **kwargs
        ):
        """Fit all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """ 

        if len(self.estimator_groups) == 0:
            raise ValueError('No estimators defined')

        group_idx = [i for i, x in enumerate(self.estimator_groups) if x['name'] == group_name]
        if len(group_idx) == 0:
            raise ValueError(f'No group named {group_name} found')

        # only iterate over the first group with the given name
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, fitting {estimator.name} estimator ')
                estimator.fit(df, *args, neptune_key=f'{group_name}_{estimator.name}', **kwargs)

        is_fitted = True
        # check if all estimators are fitted
        for group in self.estimator_groups:
            for estimator in group['estimators']:
                is_fitted = is_fitted and estimator.is_fitted

        self.is_fitted = is_fitted and len(self.estimator_groups) > 0

    def predict(
            self, 
            df : pd.DataFrame, 
            group_name : str, 
            *args, 
            **kwargs):
        
        """Predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """

        if len(self.estimator_groups) == 0:
            raise ValueError('No estimators defined')

        group_idx = [i for i, x in enumerate(self.estimator_groups) if x['name'] == group_name]
        if len(group_idx) == 0:
            raise ValueError(f'No group named {group_name} found')
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, predicting {estimator.name}')
                estimator.predict(df, inplace=True, *args, **kwargs)

    def fit_predict(
            self,
            df : pd.DataFrame,
            group_name : str,
            plot : bool = True,
        ):
        """Fit and predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        plot : bool, default=True
            If True, a plot of the calibration is generated.

        """
        self.fit(df, group_name, plot=plot)
        self.predict(df, group_name)

class OptimizationManager(BaseManager):

    def __init__(
        self,
        initial_parameters : dict,
        path : typing.Union[None, str] = None,
        load_from_file : bool = True
    ):

        logging.info('========= Initializing Optimization Manager =========')
        super().__init__(path=path, load_from_file=load_from_file)

        if not self.is_loaded_from_file:
            self.__dict__.update(initial_parameters)
            
            for key, value in initial_parameters.items():
                logging.info(f'initial parameter: {key} = {value}')
        
        logging.info('====================================================')

    def fit(self, update_dict):
        """Update the parameters dict with the values in update_dict.
        """
        self.__dict__.update(update_dict)
        self.is_fitted = True
        self.save()

    def predict(self):
        """Return the parameters dict.
        """
        return self.parameters
    
    def fit_predict(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict.
        """
        self.fit(update_dict)
        return self.predict()
    
class FDRManager(BaseManager):

    def __init__(
        self,
        feature_columns : list,
        classifier_base,
        path : typing.Union[None, str] = None,
        load_from_file : bool = True
    ):

        logging.info('========= Initializing FDR Manager =========')
        super().__init__(path=path, load_from_file=load_from_file)

        if not self.is_loaded_from_file:
            self.feature_columns = feature_columns
            self.classifier_store = {}
            self.classifier_base = classifier_base
        
        logging.info('====================================================')

    def fit_predict(
            self,
            features_df : pd.DataFrame,
            decoy_strategy : Literal['precursor', 'precursor_channel_wise', 'channel'] = 'precursor',
            competetive : bool = True,
            decoy_channel : int = -1
            ):
        """Update the parameters dict with the values in update_dict.
        """
        available_columns = list(set(features_df.columns).intersection(set(self.feature_columns)))

        # perform sanity checks
        if len(available_columns) == 0:
            raise ValueError('No feature columns found in features_df')

        if decoy_strategy == 'precursor' or decoy_strategy == 'precursor_channel_wise':
            if 'decoy' not in features_df.columns:
                raise ValueError('decoy column not found in features_df')
        
        if decoy_strategy == 'precursor_channel_wise' or decoy_strategy == 'channel':
            if 'channel' not in features_df.columns:
                raise ValueError('channel column not found in features_df')
            
        if decoy_strategy == 'channel' and decoy_channel == -1:
            raise ValueError('decoy_channel must be set if decoy_type is channel')

        if (decoy_strategy == 'precursor' or decoy_strategy == 'precursor_channel_wise')and decoy_channel > -1:
            logging.warning('decoy_channel is ignored if decoy_type is precursor')
            decoy_channel = -1

        if decoy_strategy == 'channel' and decoy_channel > -1:
            if decoy_channel not in features_df['channel'].unique():
                raise ValueError(f'decoy_channel {decoy_channel} not found in features_df')
        
        logging.info(f'performing {decoy_strategy} FDR with {len(available_columns)} features')
        logging.info(f'Decoy channel: {decoy_channel}')
        logging.info(f'Competetive: {competetive}')

        classifier = self.get_classifier(available_columns)
        if decoy_strategy == 'precursor':
            psm_df = perform_fdr(
                classifier,
                available_columns,
                features_df[features_df['decoy'] == 0].copy(),
                features_df[features_df['decoy'] == 1].copy(),
                competetive=competetive,
                group_channels = True
            )
        elif decoy_strategy == 'precursor_channel_wise':
            channels = features_df['channel'].unique()
            psm_df_list = []
            for channel in channels:
                channel_df = features_df[features_df['channel'].isin([channel, decoy_channel])].copy()
                psm_df_list.append(perform_fdr(
                    classifier,
                    available_columns,
                    channel_df[channel_df['decoy'] == 0].copy(),
                    channel_df[channel_df['decoy'] == 1].copy(),
                    competetive=competetive,
                    group_channels = True
                ))
   
            psm_df = pd.concat(psm_df_list)

        elif decoy_strategy == 'channel':
            channels = list(set(features_df['channel'].unique()) - set([decoy_channel]))
            psm_df_list = []
            for channel in channels:
                channel_df = features_df[features_df['channel'].isin([channel, decoy_channel])].copy()
                psm_df_list.append(perform_fdr(
                    classifier,
                    available_columns,
                    channel_df[channel_df['channel'] != decoy_channel].copy(),
                    channel_df[channel_df['channel'] == decoy_channel].copy(),
                    competetive=competetive,
                    group_channels = False
                ))
            
            psm_df = pd.concat(psm_df_list)
            psm_df = psm_df[psm_df['channel'] != decoy_channel].copy()
        else:
            raise ValueError(f'Invalid decoy_strategy: {decoy_strategy}')
        
        self.is_fitted = True
        self.classifier_store[column_hash(available_columns)] = classifier
        self.save()
            
        return psm_df
        
    def get_classifier(self, available_columns):
        classifier_hash = column_hash(available_columns)
        if classifier_hash in self.classifier_store:
            classifier = self.classifier_store[classifier_hash]
        else:
            classifier = sklearn.base.clone(self.classifier_base)

        if isinstance(classifier, sklearn.pipeline.Pipeline):
            for step in classifier.steps:
                if hasattr(step[1], 'warm_start'):
                    step[1].warm_start = True
                else:
                    logging.warning(f'Classifier {step[1].__class__.__name__} does not support warm_start. Will retrain classifier for each column combination.')
        else:
            if not hasattr(classifier, 'warm_start'):
                logging.warning(f'Classifier {classifier.__class__.__name__} does not support warm_start. Will retrain classifier for each column combination.')
            else:
                classifier.warm_start = True

        return classifier

    def predict(self):
        """Return the parameters dict.
        """
        raise NotImplementedError(f'predict() not implemented for {self.__class__.__name__}')
    
    def fit(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict.
        """
        raise NotImplementedError(f'fit() not implemented for {self.__class__.__name__}')
        
def column_hash(columns):
    columns.sort()
    return xxhash.xxh64_hexdigest(''.join(columns))

def perform_fdr(
        classifier, 
        available_columns,
        df_target : pd.DataFrame,
        df_decoy : pd.DataFrame,
        competetive=False,
        group_channels=True
    ):
    target_len, decoy_len = len(df_target), len(df_decoy)
    df_target.dropna(subset=available_columns, inplace=True)
    df_decoy.dropna(subset=available_columns, inplace=True)
    target_dropped, decoy_dropped = target_len - len(df_target), decoy_len - len(df_decoy)

    if target_dropped > 0:
        logging.warning(f'dropped {target_dropped} target PSMs due to missing features')

    if decoy_dropped > 0:
        logging.warning(f'dropped {decoy_dropped} decoy PSMs due to missing features')

    if np.abs(len(df_target) - len(df_decoy)) / ((len(df_target)+len(df_decoy))/2) > 0.1:
        logging.warning(f'FDR calculation for {len(df_target)} target and {len(df_decoy)} decoy PSMs')
        logging.warning(f'FDR calculation may be inaccurate as there is more than 10% difference in the number of target and decoy PSMs')

    X_target = df_target[available_columns].values
    X_decoy = df_decoy[available_columns].values
    y_target = np.zeros(len(X_target))
    y_decoy = np.ones(len(X_decoy))

    X = np.concatenate([X_target, X_decoy])
    y = np.concatenate([y_target, y_decoy])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    classifier.fit(X_train, y_train)

    psm_df = pd.concat([
        df_target,
        df_decoy
    ])

    psm_df['_decoy'] = y

    if competetive:
        group_columns = ['elution_group_idx', 'channel'] if group_channels else ['elution_group_idx']
    else:
        group_columns = ['precursor_idx']

    psm_df['proba'] = classifier.predict_proba(X)[:,1]
    psm_df.sort_values('proba', ascending=True, inplace=True)
    psm_df.reset_index(drop=True, inplace=True)
    psm_df = keep_best(psm_df, group_columns=group_columns)
    psm_df = get_q_values(psm_df, 'proba', '_decoy')

    plot_fdr(
        X_train, X_test,
        y_train, y_test,
        classifier,
        psm_df['qval']
    )
    
    return psm_df

def keep_best(df, score_column = 'proba', group_columns = ['channel', 'elution_group_idx']):
    temp_df = df.reset_index(drop=True)
    temp_df = temp_df.sort_values(score_column, ascending=True)
    temp_df = temp_df.groupby(group_columns).head(1)
    temp_df = temp_df.sort_index().reset_index(drop=True)
    return temp_df

@nb.njit
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
    _df = _df.sort_values([score_column,score_column], ascending=True)
    target_values = 1-_df[decoy_column].values
    decoy_cumsum = np.cumsum(_df[decoy_column].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum/target_cumsum
    _df['qval'] = fdr_to_q_values(fdr_values)
    return _df

def plot_fdr(
        X_train, X_test,
        y_train, y_test,
        classifier,
        qval,
    ):

    y_test_proba = classifier.predict_proba(X_test)[:,1]
    y_test_pred = np.round(y_test_proba)

    y_train_proba = classifier.predict_proba(X_train)[:,1]
    y_train_pred = np.round(y_train_proba)

    fpr_test, tpr_test, thresholds_test = sklearn.metrics.roc_curve(y_test, y_test_proba)
    fpr_train, tpr_train, thresholds_train = sklearn.metrics.roc_curve(y_train, y_train_proba)

    auc_test = sklearn.metrics.auc(fpr_test, tpr_test)
    auc_train = sklearn.metrics.auc(fpr_train, tpr_train)

    logging.info(f'Test AUC: {auc_test:.3f}')
    logging.info(f'Train AUC: {auc_train:.3f}')

    auc_difference_percent = np.abs((auc_test - auc_train) / auc_train * 100)
    logging.info(f'AUC difference: {auc_difference_percent:.2f}%')
    if auc_difference_percent > 5:
        logging.warning('AUC difference > 5%. This may indicate overfitting.')

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].plot(fpr_test, tpr_test, label=f'Test AUC: {auc_test:.3f}')
    ax[0].plot(fpr_train, tpr_train, label=f'Train AUC: {auc_train:.3f}')
    ax[0].set_xlabel('false positive rate')
    ax[0].set_ylabel('true positive rate')
    ax[0].legend()

    ax[1].hist(np.concatenate([y_test_proba[y_test == 0], y_train_proba[y_train == 0]]), bins=50, alpha=0.5, label='target')
    ax[1].hist(np.concatenate([y_test_proba[y_test == 1], y_train_proba[y_train == 1]]), bins=50, alpha=0.5, label='decoy')
    ax[1].set_xlabel('decoy score')
    ax[1].set_ylabel('precursor count')
    ax[1].legend()

    qval_plot = qval[qval < 0.05]
    ids = np.arange(0, len(qval_plot), 1)
    ax[2].plot(qval_plot, ids)
    ax[2].set_xlim(-0.001, 0.05)
    ax[2].set_xlabel('q-value')
    ax[2].set_ylabel('number of precursors')

    for axs in ax:
        # remove top and right spines
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.show()
    plt.close()
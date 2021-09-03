
import itertools
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from utils import create_logger

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier


class PSForest():
    """
    PSForest


    Example:

    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }]
    },
    """
    def __init__(
        self,
        estimators_config,
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
        folds=3,
        verbose=False
    ):
        self.mgs_instances = [
            MultiGrainedScanner(
                estimators_config['mgs'],
                stride_ratio=stride_ratio,
                folds=folds,
                verbose=verbose,
            )
            for stride_ratio in stride_ratios
        ]
        self.stride_ratios = stride_ratios

        self.c_forest = CascadeForest(estimators_config['cascade'], verbose=verbose)
        
    def fit(self, X, y):
        scanned_X = np.hstack([
            mgs.scan(X, y)
            for mgs in self.mgs_instances
        ])
        X = X.reshape(X.shape[0],-1) # 原始X转化为一维
        scanned_X = np.concatenate((scanned_X,X),axis=1)  # 加上原始特征
        self.c_forest.fit(scanned_X, y)

    def predict(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])
        X = X.reshape(X.shape[0],-1) # 原始X转化为一维
        scanned_X = np.concatenate((scanned_X,X),axis=1)  # 加上原始特征
        return self.c_forest.predict(scanned_X)
    # 不扫描
    def fit_c(self, X, y):
        X = X.reshape(X.shape[0],-1) # 原始X转化为一维
        self.c_forest.fit(X, y)

    def predict_c(self, X):
        X = X.reshape(X.shape[0],-1) # 原始X转化为一维
        return self.c_forest.predict(X)
    
    def __repr__(self):
        return '<PSForest {}>'.format(self.stride_ratios)


class MultiGrainedScanner():
    """
    Multi-Grained Scanner

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the MultiGrainedScanner.
    @param stride_ratio         The stride ratio to use for slicing the input.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """
    def __init__(
        self, estimators_config, stride_ratio=0.25, folds=3, verbose=False
    ):
        self.estimators_config = estimators_config
        self.stride_ratio = stride_ratio
        self.folds = folds

        self.estimators = [
            estimator_config['estimator_class'](**estimator_config['estimator_params'])
            for estimator_config in self.estimators_config
        ]

        self.logger = create_logger(self, verbose)
    
    
    def slices(self, X, y=None):
        """
        Given an input X with dimention N, this generates ndarrays with all the instances
        values for each window. The window shape depends on the stride_ratio attribute of
        the instance.

        For example, if the input has shape (10, 400), and the stride_ratio is 0.25, then this
        will generate 301 windows with shape (10, 100)
        """
        self.logger.debug('Slicing X with shape {}'.format(X.shape))

        n_samples = X.shape[0]
        sample_shape = X[0].shape
        window_shape = [
            max(1, int(s * self.stride_ratio)) if i < 2 else s
            for i, s in enumerate(sample_shape)
        ]

        #
        # Generates all the windows slices for X.
        # For each axis generates an array showing how the window moves on that axis.
        #
        slices = [
            [slice(i, i + window_axis) for i in range(0,(sample_axis - window_axis + 1),100)]
            for sample_axis, window_axis in zip(sample_shape, window_shape)
        ]
        total_windows = np.prod([len(s) for s in slices])

        self.logger.info('Window shape: {} Total windows: {}'.format(window_shape, total_windows))

        #
        # For each window slices, return the same slice for all the samples in X.
        # For example, if for the first window we have the slices [slice(0, 10), slice(0, 10)],
        # this generates the following slice on X:
        #   X[:, 0:10, 0:10] == X[(slice(None, slice(0, 10), slice(0, 10))]
        #
        # Since this generates on each iteration a window for all the samples, we insert the new
        # windows so that for each sample the windows are consecutive. This is done with the
        # ordering_range magic variable.
        #
        windows_slices_list = None
        ordering_range = np.arange(n_samples) + 1  # 1到m样本数

        for i, axis_slices in enumerate(itertools.product(*slices)):
            if windows_slices_list is None:
                windows_slices_list = self.pooling(X[(slice(None),) + axis_slices])
            else:
                windows_slices_list = np.insert(
                    windows_slices_list,
                    ordering_range * i,
                    self.pooling(X[(slice(None),) + axis_slices]),
                    axis=0,
                )
#                 print(windows_slices_list.shape)

        #
        # Converts any sample with dimention higher or equal than 2 to just one dimention
        #
#         windows_slices = \
#             windows_slices_list.reshape([windows_slices_list.shape[0], np.prod(window_shape)]) # 行是样本数，列是特征展开
        windows_slices = windows_slices_list.reshape(n_samples,total_windows) # 行是样本数，列是windos的个数
        #
        # If the y parameter is not None, returns the y value for each generated window
        #
#         if y is not None:
#             y = np.repeat(y, total_windows)
        print(windows_slices.shape)
        return windows_slices, y
    
    

    def scan(self, X, y=None):
        """
        Slice the input and for each window creates the estimators and save the estimators in
        self.window_estimators. Then for each window, fit the estimators with the data of all
        the samples values on that window and perform a cross_val_predict and get the predictions.
        """
        self.logger.info('Scanning and fitting for X ({}) and y ({}) started'.format(
            X.shape, None if y is None else y.shape
        ))
        self.n_classes = np.unique(y).size

        #
        # Create the estimators
        #
        sliced_X, sliced_y = self.slices(X, y)
        self.logger.debug('Slicing turned X ({}) to sliced_X ({})'.format(X.shape, sliced_X.shape))

        predictions = None
        for estimator_index, estimator in enumerate(self.estimators):
            prediction = None

            if y is None:
                self.logger.debug('Prediction with estimator #{}'.format(estimator_index))
                prediction = estimator.predict_proba(sliced_X)
            else:
                self.logger.debug(
                    'Fitting estimator #{} ({})'.format(estimator_index, estimator.__class__)
                )
                estimator.fit(sliced_X, sliced_y)

                #
                # Gets a prediction of sliced_X with shape (len(newX), n_classes).
                # The method `predict_proba` returns a vector of size n_classes.
                #
                if estimator.oob_score:
                    self.logger.debug('Using OOB decision function with estimator #{} ({})'.format(
                        estimator_index, estimator.__class__
                    ))
                    prediction = estimator.oob_decision_function_
                else:
                    self.logger.debug('Cross-validation with estimator #{} ({})'.format(
                        estimator_index, estimator.__class__
                    ))
                    prediction = cross_val_predict(
                        estimator,
                        sliced_X,
                        sliced_y,
                        cv=self.folds,
                        method='predict_proba',
                        n_jobs=-1,
                    )

            prediction = prediction.reshape((X.shape[0], -1))

            if predictions is None:
                predictions = prediction
            else:
                predictions = np.hstack([predictions, prediction])
        

        self.logger.info('Finished scan X ({}) and got predictions with shape {}'.format(
            X.shape, predictions.shape
        ))

        return predictions
    # max_pooling
    def pooling(self,array):
        pool_list =[]
        for i in array:
            pool_list.append(i.max())
        return pool_list
    
    def __repr__(self):
        return '<MultiGrainedScanner stride_ratio={}>'.format(self.stride_ratio)


class CascadeForest():
    """
    Gate-CascadeForest

    @param estimators_config    A list containing the class and parameters of the estimators for the CascadeForest.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """
    def __init__(self, estimators_config, folds=3, verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds

        self.logger = create_logger(self, verbose)
        self.err_max = []
        self.final_C = XGBClassifier(n_estimators=500)
    def fit(self, X, y):
        self.logger.info('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None

        while True:
            self.logger.info('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]
            # 得到每级的预测list
            predictions = []
            oob_scores = []
            for estimator in estimators:
                self.logger.debug('Fitting X ({}) and y ({}) with estimator {}'.format(
                    X.shape, y.shape, estimator
                ))
                estimator.fit(X, y)

                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )
                # 袋外准确度
                
                oob_score = estimator.oob_score_
                oob_scores.append(oob_score)
                
                predictions.append(prediction)
            print(oob_scores)
            self.logger.info('Level {}:: got all predictions'.format(self.level + 1))
            # 过滤掉袋外正确率最少的predict
            for i in range(2):
                del predictions[oob_scores.index(min(oob_scores))]
                del oob_scores[oob_scores.index(min(oob_scores))]
                self.err_max.append(oob_scores.index(min(oob_scores)))
            print(self.err_max)
            

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            # 预测的类别
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )
            # 当前层的正确率
            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if (self.max_score is None or score > self.max_score):
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
                #
                # Stacks horizontally the predictions to each of the samples in X
                #
                X = np.hstack([X] + predictions)
            else:
                break
#         self.final_C.fit(X,y)
        
    def predict(self, X):

        for estimators in self.levels:

            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            oob_scores = [
                estimator.oob_score_
                for estimator in estimators
            ]
            print(oob_scores)
            
            self.logger.info('Shape of predictions: {} shape of X: {}'.format(
                np.array(predictions).shape, X.shape
            ))
            # 过滤掉袋外正确率最少的几个predict
            for i in range(2):
                del predictions[oob_scores.index(min(oob_scores))]
                del oob_scores[oob_scores.index(min(oob_scores))]
                
            X = np.hstack([X] + predictions)
#         print(self.final_C.predict(X))
#         return self.final_C.predict(X)
        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

    def __repr__(self):
        return '<Gate-CascadeForest forests={}>'.format(len(self.estimators_config))

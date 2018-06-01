from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

from collections import Counter

import crowdai

# import module with the functions to calculate technical indicators
import ieee.tech_indicators as ti

import pandas as pd
import numpy as np
import scipy.stats as stat
import os
import pickle
import copy
import operator
import dask.dataframe as dd
import warnings
warnings.filterwarnings('ignore')


class PrepareDataset:
    def __init__(self, path_to_data,
                 dataset_name='full_dataset.csv'):
        self.path_to_data = path_to_data
        self.dataset_name = dataset_name
        # self.pred_template_name = pred_template_name
        self.initial_df = pd.read_csv(os.path.join(self.path_to_data, self.dataset_name))
        # self.pred_template = pd.read_csv(os.path.join(self.path_to_data, self.pred_template_name))
        self.time_periods = np.unique(self.initial_df['time_period'], return_index=True)[0]
        self.time_periods_index = np.unique(self.initial_df['time_period'], return_index=True)[1]
        self.transformed_df = None
        self.scaler = None
        self.transformed_scaled_df = None
        self.pca = None

    def collapse_basic_indicators(self, df=None, indicators_range=range(1, 71)):
        if df is None:
            df = copy.copy(self.initial_df)

        print('Begin to collapse basic indicators')
        print('{} indicators were selected'.format(indicators_range))

        variable_list = ["X" + str(i) + '_' for i in indicators_range]

        for var in variable_list:
            df[var + 'avg'] = df.filter(regex=var).mean(axis=1)

        for var in variable_list:
            df[var + 'std'] = df.filter(regex=var).std(axis=1)

        for var in variable_list:
            df[var + 'avg' + '_pctile'] = stat.rankdata(df[var + 'avg']) / df[var + 'avg'].shape[0]

        for var in variable_list:
            df[var + 'std' + '_pctile'] = stat.rankdata(df[var + 'std']) / df[var + 'std'].shape[0]

        model_data_new = pd.concat([df.iloc[:, 0:5],
                                    df.filter(regex='avg'),
                                    df.filter(regex='std'),
                                    ], axis=1)

        model_data_new['is_second_half'] = model_data_new['time_period'].apply(lambda x: 1 if x.endswith('1') else 0)

        model_data_new.fillna(0, inplace=True)

        self.transformed_df = model_data_new
        print('Basic indicators collapsed')

    def drop_outliers(self, quantile = None):
        if quantile is None:
            quantile = 0.001

        print('Start to drop rows with outliers. {} quantile will be removed from each side'.format(quantile/2.))
        idx_to_drop = []
        features_list = list(self.transformed_df.filter(regex='avg$').columns)

        for col in features_list:
            condition1 = self.transformed_df.Train == 1
            condition2 = self.transformed_df[col] > self.transformed_df[col].quantile(1. - quantile/2.)
            condition3 = self.transformed_df[col] < self.transformed_df[col].quantile(quantile/2.)
            to_drop = list(self.transformed_df[(condition1) & (condition2)].index)
            idx_to_drop += to_drop
            to_drop = list(self.transformed_df[(condition1) & (condition3)].index)
            idx_to_drop += to_drop
        self.transformed_df = self.transformed_df.drop(list(set(idx_to_drop)))
        print('Done. {} rows were removed'.format(len(list(set(idx_to_drop)))))

    def add_tech_indicators(self, df=None):
        if df is None:
            df = self.transformed_df if self.transformed_df is not None else self.initial_df

        print('Start to form basic indicators')
        data_agg_avg = df.groupby(['time_period']).mean()
        period_agg_df = pd.DataFrame(index=self.time_periods)
        period_agg_df['Close'] = data_agg_avg['Norm_Ret_F6M']

        # Shift Norm_Ret_F6M to the next period to avoid future looking
        period_agg_df['Close'] = period_agg_df['Close'].shift(1)
        period_agg_df['Close'].fillna(method='bfill', inplace=True)

        period_agg_df = ti.MA(period_agg_df, 2)
        period_agg_df = ti.MA(period_agg_df, 3)

        period_agg_df = ti.EMA(period_agg_df, 2)
        period_agg_df = ti.EMA(period_agg_df, 3)

        period_agg_df = ti.MOM(period_agg_df, 2)
        period_agg_df = ti.MOM(period_agg_df, 3)

        period_agg_df = ti.ROC(period_agg_df, 2)
        period_agg_df = ti.ROC(period_agg_df, 3)

        period_agg_df = ti.MACD(period_agg_df, 2, 3)

        period_agg_df = ti.KST(period_agg_df, 1, 2, 3, 4, 1, 2, 3, 4)

        period_agg_df = ti.TSI(period_agg_df, 2, 2)

        period_agg_df = ti.COPP(period_agg_df, 2)
        period_agg_df = ti.COPP(period_agg_df, 3)

        period_agg_df = ti.STDDEV(period_agg_df, 2)
        period_agg_df = ti.STDDEV(period_agg_df, 3)

        # concatenate technical indicator with the transformed dataset
        df = df.join(period_agg_df, on='time_period')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        self.transformed_df = df
        print('Done')

    def generate_synthetic_indicators(self, types=None):
        features_list = list(self.transformed_df.filter(regex='avg$').columns)

        if types is None:
            types = ['substract', 'multiply']

        for i1, col1 in enumerate(features_list):
            print('Processing column {}'.format(col1))
            for i2, col2 in enumerate(features_list):
                if 'substract' in types:
                    self.transformed_df['%s_%s_1' % (col1, col2)] = self.transformed_df[col1] - \
                                                                    self.transformed_df[col2]
                if 'add' in types:
                    self.transformed_df['%s_%s_2' % (col1, col2)] = self.transformed_df[col1] + \
                                                                    self.transformed_df[col2]

                if 'divide' in types:
                    self.transformed_df['%s_%s_3' % (col1, col2)] = self.transformed_df[col1] / \
                                                                    (self.transformed_df[col2]+0.01)

                if 'multiply' in types:
                    self.transformed_df['%s_%s_4' % (col1, col2)] = self.transformed_df[col1] * \
                                                                    self.transformed_df[col2]


        print('Done')

    def scale_df(self, scaled_columns=None):
        if scaled_columns is None:
            scaled_columns = [x for x in self.transformed_df.columns if x.endswith('pctile')]
            columns_for_scale = set(self.transformed_df.iloc[:, 5:].columns) - set(scaled_columns)
            self.scaler = MinMaxScaler()
            df_for_scale = copy.copy(self.transformed_df)
            df_for_scale.loc[:, columns_for_scale] = self.scaler.fit_transform(df_for_scale.loc[:, columns_for_scale])
            self.transformed_scaled_df = df_for_scale

    def apply_pca_to_scaled_df(self, n_components=0.99, svd_solver='full'):
        self.pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=17)
        self.pca.fit(self.transformed_scaled_df.iloc[:, 5:])
        pca_arr = self.pca.transform(self.transformed_scaled_df.iloc[:, 5:])
        columns_pca = ['pca_' + str(x) for x in range(1, pca_arr.shape[1]+1)]
        pca_df = pd.DataFrame(pca_arr, columns=columns_pca)
        self.transformed_scaled_df = pd.concat([self.transformed_scaled_df.iloc[:, :5], pca_df], axis=1)




class OptimizeRidgeHyperparameters:
    def __init__(self, df, time_periods, time_periods_index, feat_selection_filename,
                 model_tune_filename, path_to_optim='../optimization', ):
        self.df = df
        self.time_periods = time_periods
        self.time_periods_index = time_periods_index
        self.path_to_optim = path_to_optim
        self.feat_selection_filename = feat_selection_filename
        self.model_tune_filename = model_tune_filename
        self.feat_selection_res = None
        self.model_tune_res = None

    def calc_metrics(self, time_period, predicted_rank):
        # subset actual values for prediction time_period
        actuals = self.df.loc[(self.df['time_period'] == time_period) & (self.df['Train'] == 1), :]

        # join predictions onto actuals
        actuals['Rank_F6M_pred'] = predicted_rank

        # calculate spearman correlation
        spearman = stat.spearmanr(actuals['Rank_F6M'], actuals['Rank_F6M_pred'])[0]

        # calculate NDCG = DCG of Top 20% / Ideal DCG of Top 20%
        # subset top 20% predictions
        t20 = actuals.loc[actuals['Rank_F6M_pred'] <= np.nanpercentile(actuals['Rank_F6M_pred'], 20), :]
        t20['discount'] = np.amax(actuals['Rank_F6M_pred']) / (
                    np.amax(actuals['Rank_F6M_pred']) + actuals['Rank_F6M_pred'])
        t20['gain'] = t20['Norm_Ret_F6M'] * t20['discount']
        dcg = np.sum(t20['gain'])

        # subset top 20% actuals
        i20 = actuals.loc[actuals['Rank_F6M'] <= np.nanpercentile(actuals['Rank_F6M'], 20), :]
        i20['discount'] = np.amax(actuals['Rank_F6M']) / (np.amax(actuals['Rank_F6M']) + actuals['Rank_F6M'])
        i20['gain'] = i20['Norm_Ret_F6M'] * i20['discount']
        idcg = np.sum(i20['gain'])

        ndcg = dcg / idcg

        # return time_period, spearman correlation, NDCG
        return pd.DataFrame([(time_period, spearman, ndcg)], columns=['time_period', 'spearman', 'NDCG'])

    # custom score function for RFE selector
    @staticmethod
    def spearman_scorer(model, x_array, y_true):
        y_pred = model.predict(x_array)
        true_rank = len(y_true) - stat.rankdata(y_true, method='ordinal').astype(int) + 1
        rank_pred = len(y_pred) - stat.rankdata(y_pred, method='ordinal').astype(int) + 1
        spearman = stat.spearmanr(true_rank, rank_pred)[0]
        return spearman

    # custom CV fold creator for RFE selector
    def create_cvs(self, train_start_period, prediction_periods):
        cv_splits = []
        for time in prediction_periods:
            train_window_start = self.time_periods_index[self.time_periods == train_start_period][0]
            train_window_end = self.time_periods_index[self.time_periods == time][0]
            train_cv_model_data = self.df.iloc[range(train_window_start, train_window_end), :]
            train_idx = train_cv_model_data.loc[train_cv_model_data['Train'] == 1].index
            valid_idx = self.df.loc[(self.df['time_period'] == time) & (self.df['Train'] == 1)].index
            cv_splits.append((np.array(train_idx), np.array(valid_idx)))
        return cv_splits

    @staticmethod
    def fit_ridge(dask_df, per, best_cols, alpha=1.):
        ridge_model_data = dask_df[dask_df.time_period.isin(per)]
        ridge = Ridge(alpha=alpha, random_state=17)
        ridge.fit(ridge_model_data.loc[ridge_model_data['Train'] == 1, best_cols],
                  ridge_model_data.loc[ridge_model_data['Train'] == 1, 'Norm_Ret_F6M'])
        return ridge

    def measure_predict(self, dask_df, train_start_period, prediction_period, best_cols):
        train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])
        s_ind = list(self.time_periods).index(train_start_period)
        e_ind = list(self.time_periods).index(prediction_period)
        per = self.time_periods[s_ind:e_ind]
        ridge = self.fit_ridge(dask_df, per, best_cols)
        time = prediction_period

        if time != '2017_1':
            train_predictions = ridge.predict(
                dask_df.loc[(dask_df['time_period'] == time) &
                            (dask_df['Train'] == 1), best_cols])

            train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions, method='ordinal').astype(
                int) + 1

            train_results = train_results.append(self.calc_metrics(time_period=time,
                                                                   predicted_rank=train_rank_predictions))

        return train_results['spearman'].mean()

    def select_best_features(self, step=10):

        model_results = pd.DataFrame(
            columns=['start_train_period', 'predict_period', 'model', 'RFE_step', 'best_cols', 'spearman'])

        cols = self.df.iloc[:, 5:].columns

        # create dask df to avoid the problem with large dataset in memory
        dd_df = dd.from_pandas(self.df, npartitions=2, chunksize=None)

        k = list(range(11, 42))

        for ind, ppi in enumerate(k[:-1]):
            prediction_period = self.time_periods[ppi]
            print('Start calculations for {}'.format(prediction_period))

            for spi in range(ppi):
                train_start_period = self.time_periods[spi]
                print('Start calculations with first training period equals {}'.format(train_start_period))
                ridge_for_selection = Ridge(random_state=17)
                cv_splits = self.create_cvs(train_start_period, [prediction_period])
                selector_ridge = RFECV(ridge_for_selection, cv=cv_splits, step=step,
                                       scoring=self.spearman_scorer)
                selector_ridge.fit(self.df.loc[:, cols], self.df['Norm_Ret_F6M'])
                best_cols_with_selector = np.asarray(cols)[selector_ridge.support_]
                print('Best columns are {}'.format(best_cols_with_selector))
                spearman_score = self.measure_predict(dd_df, train_start_period,
                                                      prediction_period, best_cols_with_selector)
                print('Spearman score is {}'.format(spearman_score))
                model_results = model_results.append({'start_train_period': train_start_period,
                                                      'predict_period': prediction_period,
                                                      'model': selector_ridge.estimator,
                                                      'RFE_step': selector_ridge.step,
                                                      'best_cols': best_cols_with_selector,
                                                      'spearman': spearman_score},
                                                     ignore_index=True)

                # write current results to file and class property
                with open(os.path.join(self.path_to_optim, self.feat_selection_filename), 'wb') as f:
                    pickle.dump(model_results, f)
                self.feat_selection_res = model_results

    def load_best_features_from_file(self, filename=None):
        if filename is None:
            filename = self.feat_selection_filename

        with open(os.path.join(self.path_to_optim, filename), 'rb') as f:
            temp_res = pickle.load(f)

        # dirty check that loaded results have right object type and not empty
        if len(temp_res['start_train_period'].values[0]) > 0:
            self.feat_selection_res = temp_res

        return None

    def get_best_features_df(self):
        if self.feat_selection_res is not None:
            gmr = self.feat_selection_res.groupby('predict_period')
            return gmr.apply(lambda g: g[g['spearman'] == g['spearman'].max()])

    def select_best_hp_with_best_features(self, alpha_ridge=None):
        if alpha_ridge is None:
            alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

        best_params_df = self.get_best_features_df()

        model_results_alpha = pd.DataFrame(
            columns=['start_train_period', 'predict_period', 'model', 'best_cols', 'spearman'])

        # create dask df to avoid the problem with large dataset in memory
        dd_df = dd.from_pandas(self.df, npartitions=2, chunksize=None)

        for time in self.time_periods[11:-1]:
            train_start_period = \
                best_params_df.loc[best_params_df.index.get_level_values('predict_period') == time][
                    'start_train_period'].values[0]
            best_cols = \
                best_params_df.loc[best_params_df.index.get_level_values('predict_period') == time]['best_cols'].values[
                    0]

            s_ind = list(self.time_periods).index(train_start_period)
            e_ind = list(self.time_periods).index(time)
            per = self.time_periods[s_ind:e_ind]

            for alpha in alpha_ridge:

                ridge = self.fit_ridge(dask_df=dd_df, per=per, best_cols=best_cols, alpha=alpha)

                train_predictions = ridge.predict(dd_df.loc[(dd_df['time_period'] == time) & (dd_df['Train'] == 1),
                                                  best_cols])

                train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions,
                                                                                method='ordinal').astype(int) + 1

                spearman_score = self.calc_metrics(time_period=time,
                                                   predicted_rank=train_rank_predictions)['spearman'].values[0]

                model_results_alpha = model_results_alpha.append({'start_train_period': train_start_period,
                                                                  'predict_period': time,
                                                                  'model': ridge,
                                                                  'best_cols': best_cols,
                                                                  'spearman': spearman_score},
                                                                 ignore_index=True)

                # write current results to file and class property
                with open(os.path.join(self.path_to_optim, self.model_tune_filename), 'wb') as f:
                    pickle.dump(model_results_alpha, f)

                self.model_tune_res = model_results_alpha

            print("Time period " + time + " completed.")

    def get_best_model_df(self):
        if self.model_tune_res is not None:
            gmr = self.model_tune_res.groupby('predict_period')
            return gmr.apply(lambda g: g[g['spearman'] == g['spearman'].max()])
        return None

    def get_most_frequent_feat_indices(self, threshold=11):
        glob_hyper_pos = []
        best_model_df = self.get_best_model_df()
        for time in best_model_df.predict_period.values:
                glob_hyper_pos += list(best_model_df.loc[best_model_df.predict_period == time]['best_cols'].values[0])

        occur_pos = Counter(glob_hyper_pos)
        sorted_occur_pos = sorted(occur_pos.items(), key=operator.itemgetter(1), reverse=True)

        freq_col_list_pos = [x for (x, y) in sorted_occur_pos if y >= int(threshold)]
        xs_pos = [int(x[1:3]) if x[2] != '_' else int(x[1:2]) for x in freq_col_list_pos]
        return list(set(xs_pos))


class FormRidgePredictions:
    def __init__(self, df, time_periods, time_periods_index, best_params_dict,
                 ):
        self.df = df
        self.time_periods = time_periods
        self.time_periods_index = time_periods_index
        self.best_params_dict = best_params_dict
        self.valid_res = None
        self.test_predictions = {}

    def calc_metrics(self, time_period, predicted_rank):
        # subset actual values for prediction time_period
        actuals = self.df.loc[(self.df['time_period'] == time_period) & (self.df['Train'] == 1), :]

        # join predictions onto actuals
        actuals['Rank_F6M_pred'] = predicted_rank

        # calculate spearman correlation
        spearman = stat.spearmanr(actuals['Rank_F6M'], actuals['Rank_F6M_pred'])[0]

        # calculate NDCG = DCG of Top 20% / Ideal DCG of Top 20%
        # subset top 20% predictions
        t20 = actuals.loc[actuals['Rank_F6M_pred'] <= np.nanpercentile(actuals['Rank_F6M_pred'], 20), :]
        t20['discount'] = np.amax(actuals['Rank_F6M_pred']) / (
                np.amax(actuals['Rank_F6M_pred']) + actuals['Rank_F6M_pred'])
        t20['gain'] = t20['Norm_Ret_F6M'] * t20['discount']
        dcg = np.sum(t20['gain'])

        # subset top 20% actuals
        i20 = actuals.loc[actuals['Rank_F6M'] <= np.nanpercentile(actuals['Rank_F6M'], 20), :]
        i20['discount'] = np.amax(actuals['Rank_F6M']) / (np.amax(actuals['Rank_F6M']) + actuals['Rank_F6M'])
        i20['gain'] = i20['Norm_Ret_F6M'] * i20['discount']
        idcg = np.sum(i20['gain'])

        ndcg = dcg / idcg

        # return time_period, spearman correlation, NDCG
        return pd.DataFrame([(time_period, spearman, ndcg)], columns=['time_period', 'spearman', 'NDCG'])

    def fit_ridge(self, train_start_period, prediction_period, best_cols, model):

        train_window_start = self.time_periods_index[self.time_periods == train_start_period][0]
        train_window_end = self.time_periods_index[self.time_periods == prediction_period][0]

        ridge_model_data = self.df.loc[range(train_window_start, train_window_end), :]
        ridge_model_data = ridge_model_data[ridge_model_data['time_period'].notnull()]

        ridge = model

        # fit using training data only (Train == 1)
        ridge.fit(ridge_model_data.loc[ridge_model_data['Train'] == 1, best_cols],
                  ridge_model_data.loc[ridge_model_data['Train'] == 1, 'Norm_Ret_F6M'])

        return ridge

    def predict_ridge(self):
        train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])

        for time in self.time_periods[11:]:
            period_from_best_model = '2016_2' if time == '2017_1' else time

            train_start_period = \
                self.best_params_dict.loc[self.best_params_dict.index.get_level_values('predict_period') ==
                                          period_from_best_model]['start_train_period'].values[0]
            best_cols = \
                    self.best_params_dict.loc[self.best_params_dict.index.get_level_values('predict_period') ==
                                              period_from_best_model]['best_cols'].values[0]
            best_model = \
                    self.best_params_dict.loc[self.best_params_dict.index.get_level_values('predict_period') ==
                                              period_from_best_model]['model'].values[0]

            ridge = self.fit_ridge(train_start_period=train_start_period, prediction_period=time,
                                   best_cols=best_cols, model=best_model)

            if time != '2017_1':
                train_predictions = ridge.predict(self.df.loc[(self.df['time_period'] == time) &
                                                              (self.df['Train'] == 1), best_cols])

                train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions,
                                                                                method='ordinal').astype(int) + 1

                train_results = train_results.append(self.calc_metrics(time_period=time,
                                                                       predicted_rank=train_rank_predictions))

            test_predictions = ridge.predict(self.df.loc[(self.df['time_period'] == time) & (self.df['Train'] == 0),
                                                         best_cols])

            self.test_predictions[time] = list(test_predictions)
            print("Time period " + time + " completed.")

        self.valid_res = train_results


class PrepareSubmitFile:
    def __init__(self, model_list, path_to_data, path_to_results,
                 pred_template_name='prediction_template.csv', result_file_name='ridge_ensemble.csv'):
        self.model_list = model_list
        self.pred_template_name = pred_template_name
        self.path_to_data = path_to_data
        self.path_to_results = path_to_results
        self.result_file_name = result_file_name
        self.best_predictors_df = None

    def select_predictor_for_each_period(self):
        df_best = pd.DataFrame(columns=['time_period', 'best_value', 'best_model_index'])
        for period in self.model_list[0].valid_res['time_period'].values:
            b_val = -999
            b_ind = None
            for c_ind, model in enumerate(self.model_list):
                c_val = (model.valid_res.loc[model.valid_res.time_period == period]['spearman'].values +
                         model.valid_res.loc[model.valid_res.time_period == period]['NDCG'].values) / 2.
                if c_val > b_val:
                    b_val = c_val
                    b_ind = c_ind
            df_best = df_best.append({'time_period': period,
                                      'best_value': b_val[0],
                                      'best_model_index': b_ind},
                                     ignore_index=True)

        # for the period 2017_1 we select the best model for the period 2016_2
        last_period_best = df_best[-1:].values[0]

        df_best = df_best.append({'time_period': '2017_1',
                                      'best_value': last_period_best[1],
                                      'best_model_index': last_period_best[2]},
                                     ignore_index=True)

        self.best_predictors_df = df_best

    def form_file(self):
        pred_template = pd.read_csv(os.path.join(self.path_to_data, self.pred_template_name))
        for period in self.best_predictors_df['time_period'].values:
            best_model_index = self.best_predictors_df.loc[self.best_predictors_df.time_period == period,
                                                           'best_model_index'].values[0]
            test_predictions = self.model_list[best_model_index].test_predictions[period]
            test_rank_predictions = len(test_predictions) - stat.rankdata(test_predictions, method='ordinal').astype(
                int) + 1
            pred_template.loc[pred_template['time_period'] == period, 'Rank_F6M'] = test_rank_predictions

        pred_template.to_csv(os.path.join(self.path_to_results, self.result_file_name))

    def submit_results(self, api_key="8229d44990cdd9496c4ae53cc9308aed"):
        challenge = crowdai.Challenge("IEEEInvestmentRankingChallenge", api_key)
        result = challenge.submit(os.path.join(self.path_to_results, self.result_file_name), round=2)
        print(result)

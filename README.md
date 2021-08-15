# Predict Future Sales

This repo contains my solution for the [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) Kaggle In-Class competition which is a final project for ["How to win a data science competition" Coursera course](https://www.coursera.org/learn/competitive-data-science).

The task was to predict the number of sales for the set of items in different shops for November 2015. Given dataset consisted of sales history of the previous 3 years: date, number of items sold, item price, item, category, shop (watch [Kaggle](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data) for details)

Performed exploratory data analysis, preprocessing & feature engineering and predicting with [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) + [Optuna](https://optuna.readthedocs.io/en/stable/index.html) for hyperparameter tuning. Achieved RMSE score: **0.953814** on public and **0.946986** on private leaderboard (10/10 on Coursera).

All the work was done in a Kaggle Kernels.

## Data Exploration • [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/predict-future-sales/blob/main/eda.ipynb)

As for any other data science task, I started with exhaustive exploratory data analysis and build a lot of plots to get some intuition about the given data.

It helps a lot with the construction of a validation scheme and gives some ideas about distribution change over time, and the impact of particular categories and shops on the number of sales. Additionally, a persistent ("last month") prediction was created, which gives a baseline of 1.16777 rmse on a public LB.

Also, I suppose that there are no leaks in the test data because it was constructed in the most "secure" way: items & shops id's given and nothing more.

## Validation scheme and construction of a training dataset

The most important observation about test data - consists of all the possible item-shop pairs for the given set of items. Two obvious conclusions:

1. Validation dataset should be constructed in the same way
2. Noticing that sales history doesn't contain such a variety of item-shop pairs, it's easy to suppose that a high number of values will be a zero. That means we need to give our machine learning model the possibility to learn these zeros.

Together with almost the same set of shops every month and a similar number of totally new items per month in both sales history and test set, I found the most appropriate way to construct a training dataset as a Cartesian product of unique item IDs and shop IDs for each month starting from Jan 2014.

For each month, the percentage of pairs that appears in the previous month was similar (see EDA notebook for details), then probably any of the months can be used as a validation set (of course we should make a split respecting the time). I choose a hold-out validation scheme for less RAM and time consumption on a big dataset and left October 2014 (date_block_num=33) for validation.

## Feature Engineering • [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/predict-future-sales/blob/main/submission/feature-engineering.ipynb)

Feature engineering is always the most important step in the solution of a data science task. And we have time in our data, so we should respect it when constructing the features to avoid using information from the future, which will be impossible for the test set.

The next features were created (the same names as in notebooks):

- **Categorical**
  - `item_id` - in the original form, treated as a numeric by GBDT because of high cardinality (you can think about it as additional items grouping just by id).
  - `item_category_id` - in the original form.
  - `primary_category_id` - extracted the more general category from item category name - the first half before the dash, then encode it in order of appearance.
  - `shop_id` - in original form.
  - `city_id` - extracted city info from the shop name - the first word, then encode it in order of appearance.
  - `month_num` - month number in year, from 0 to 11.
- **Aggregated stats of previous sales**
  - `lag_*_target_sum` - target (monthly sum of item-shop sales) 1, 2, 3, and 12 lags.
  - `lag_*_item_sales_mean` - average values of target grouped by item, 1, 2, 3, and 12 lags.
  - `lag_*_item_shop_mean` - average values of target grouped by shop, 1, 2, 3, and 12 lags.
  - `lag_*_item_city_mean` - average values of target grouped by city, 1, 2, 3, and 12 lags.
  - `lag_*_item_cat_mean` - average values of target grouped by category, 1, 2, 3, and 12 lags.
  - `lag_*_item_primarycat_mean` - average values of target grouped by primary category, 1, 2, 3, and 12 lags.
  - `lag_*_item_shop_cat_mean` - average values of target grouped by shop and category, 1, 2, 3, and 12 lags.
  - `lag_*_item_shop_primarycat_mean` - average values of target grouped by shop and primary category, 1, 2, 3, and 12 lags.
   - _All missing values are filled with zero_
- **Prices**. Because prices change over time, I decide to model its trend with linear regression and divide the monthly average price of an item by this month's trend, to get values marks if an item is more expensive than average or cheaper. Notice that all the training data was used for regression fitting, and it somewhat distorts the validation process, because do not respect the time
  - `lag_*_item_id_relative_price` - as described above,  1, 2, and 3 lags.
  - _All missing values are filled with average by the category_
- **Times**. Time since particular event often can be a good feature for the time-distributed data. In this case, these features significantly improved the result.
  - `time_from_last_shop_item_sale` - number of months elapsed since the last item sale in the same shop. _Missing values are filled with outlier (231)_.
  - `time_from_first_shop_item_sale` - number of months elapsed since the first item sale in the same shop. _Missing values are filled with outlier (231)_.
  - `num_months_with_shop_item_sales` - number of months between `time_from_last_shop_item_sale` and `time_from_first_shop_item_sale` with sales of this item in the same shop. _Missing values are filled with 0_.
  - `time_from_last_item_sale` - number of months elapsed since the last (item sale in all shops). _Missing values are filled with outlier (231)_.
  - `time_from_first_item_sale` - number of months elapsed since the first item sale (in all shops). _Missing values are filled with outlier (231)_.
  - `num_months_with_item_sales` - number of months between `time_from_last_item_sale` and `time_from_first_item_sale` with sales of this item. _Missing values filled are with 0_.
- **Weekends**. I noticed that some items are selling on weekends more often than others, and tried to construct a new feature based on it.
  - `weekend_weight` - the ratio of item weekend sales in all the previous months (_missing values are filled with average by category_), multiplied by the number of weekends in the current one.
- **Combinations**. For all the lagged features (including relative price) create some additional columns 
  - `ratio_1_to_2_*` - ratio of first feature lag to second.
  - `ratio_2_to_3_*` - ratio of second feature lag to third.
  - `last_3m_avg_*` - average of 1, 2 and 3 lags.

Total: 76 features.

## Modeling (LightGBM) • [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/predict-future-sales/blob/main/submission/lightgbm.ipynb)

### Why GBDT?
1. First point is the computational efficiency on the big dataset, with limitation of Kaggle kernels, model should trains and predicts fast enough to make exploring usefulness of created features and hyperparameter tuning possible. There is no much options:
    - GBDT (on GPU)
    - Neural Networks (on GPU)
    - Linear Models (with Vowpal Wabbit)

2. True target values on a leaderboard are clipped into the [0, 20] range, and the metric - RMSE is sensitive to big errors.

    In terms of fitting to leaderboard (this is competition anyway), it means, that for any high value (>20) we don't care how close we estimate it if we predict more than 20, but for rmse it's not the case: while optimizing this metric model more likely will be biased to the high values which have a more wide range instead of perfect fitting the values in a relatively small range of [0, 20].

    To overcome this problem we can use sample weights to reduce the impact of high values, somewhat modify the loss function to take this into account, or which is much more simple: clip our train and validation target into [0, 20] too.
  
    But from this comes the problem of "clustering" high values at the end of this segment, which is producing non-linear relationship: toy example - if in the previous month were 60 sales - the model should predict 20 if 25 sales - it should predict 20 too, but, maybe, be less confident in this prediction. This is not the case for the linear models, NeuralNets probably can learn this relationship if deep enough, but increasing depth usually leads to overfitting + there are a lot of categorical variables, which are hard to deal with both for linear models and neural networks. At the same time, decision trees can deal easily with such non-linear relationships and categorical variables.

Also, a solution can be built with a bunch of simple models for each item or item-shop pair: autoregressions, exponential smoothings, additive models like Facebook Prophet, etc. This approach's pitfalls are hyperparameter tuning (if there) and predicting new items sales - maybe additional models for categories should be created. With all of this, this approach seems to be very complicated and requires a lot of manual work to be a full-fledged solution, but, maybe it can take place in stacking.

### Why LightGBM?

LightGBM (using the train API) is a nice choice in terms of RAM consumption and training speed, and it has great Optuna integration for hyperparameter tuning.

In this situation, XGBoost uses much more RAM for its DMatrix and creates memory leaks which quickly become critical with the limitation of Kaggle Kernel, while LightGBM doesn't use too much and seems to support 32 and 16-bit numbers, which allows running multiple trainings within a single kernel session. Additionally, LightGBM supports categorical features out-of-box which is a nice point too.

Also, CatBoost can work with pandas dataframes and numpy arrays natively, use a bunch of advanced techniques to deal with categorical data, and have a nice sklearn-compatible API as default. I tried CatBoostRegressor in the early stages, but categorical features took too much time to process and it was easily outperformed by LightGBM (when using CatBoost symmetric trees growing policy, I didn't test another. Maybe, leafwise "LightGBM-style" growing will work well too). And, unfortunately, CatBoost still doesn't support cross-validation on GPU.

### Modeling & hyperparameter tuning

Features were created specifically for GBDT, so no additional processing is required. The only thing I do - downcast numeric columns to float16 for reducing RAM usage.

In general, models aren't really stable on cross-validation tests, std of validation scores on last three months was high, and score on validation set was always better than on the leaderboard, but values have been correlated and reducing validation score always led to reducing LB score.

Also, I don't retrain the model on all the training data after validation, because it for some reason leads to overfitting and reducing scores on the leaderboard. Anyway, looks like created features bring enough information to reduce the time impact, so the prediction in one month looks OK too.

Hyperparameter was tuned with [Optuna's LightGBMTuner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.LightGBMTuner.html#optuna.integration.lightgbm.LightGBMTuner) which stepwise finds best values of  `feature_fraction` → `num_leaves` → `bagging` → `feature_fraction` (again) → `regularization_factors` (L1, L2) → `min_data_in_leaf`. It works well, found parameters were better that ones I found by manual tuning.

Resulting parameters:
```python
{
    'device_type': 'gpu',
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'feature_pre_filter': False,
    'lambda_l1': 2.5361081166471375e-07,
    'lambda_l2': 2.5348407664333426e-07,
    'num_leaves': 188,
    'feature_fraction': 0.6,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 50
}
```
This set of parameters already gives  ~0.905 validation rmse after ~230 iterations, but maybe it can be slightly improved by tuning learning rate.

## Ways for improvement

- **Validation**. I think the validation scheme can be improved to correlate better with the leaderboard. Then validation score will better represent the quality of a model, and the better one can be chosen. It will be better to use cross-validation on the last few months to avoid multiple comparison overfitting problem.
- **Prices**. For now, prices aren't top features by importance, but I believe it is possible to extract more useful information from it. Also, current prices cols should be reconstructed with respect to time. Maybe it will be a good idea to use lagged sales number of items in the same price group.
- **Feature selection**. Some features are almost useless in terms of the number of splits and gain, especially lag2 to lag3 ratios and shop/city sales.
- **Data cleaning**. Looks like particular shops and cities history do not impact the number of sales. So maybe deletion of shops that don't appear in the test set can improve validation/score.
- **Model explanation**. Built trees can be analyzed for insights about feature interactions. And packages like [SHAP](https://shap.readthedocs.io/en/latest/index.html) can be used to get more understanding about data impact on prediction.
- **Other preprocessing, models, and ensembling**. I believe, the problem described in `Why GBDT?` can be overcome by clipping all the target values to [0, 20]. Or maybe linear regression will be good already? Maybe GBDT on clipped values will give a better score? (looks like it will) What about LSTM networks? There is a lot of new things to try, and even if it doesn't give a better result in terms of a one-model solution, it probably will introduce new information that can be used with stacking ensembles.

## Some other great kernels for inspiration
- [deepdivelm's Feature Engineering/LightGBM/Exploring Performance](https://www.kaggle.com/deepdivelm/feature-engineering-lightgbm-exploring-performance/)
- [dlarionov's Feature engineering, xgboost](https://www.kaggle.com/dlarionov/feature-engineering-xgboost)
- [werooring's 🚩 (TOP 3.5%) LightGBM with Feature Engineering 🤗]( https://www.kaggle.com/werooring/top-3-5-lightgbm-with-feature-engineering)




If you had read, or at least scroll it all, thank you for your attention!!!
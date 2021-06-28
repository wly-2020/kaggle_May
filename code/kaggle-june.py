#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import tensorflow as tf
import random
import os

os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED = 40

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

label_map = {
    'Class_1': 0,
    'Class_2': 1,
    'Class_3': 2,
    'Class_4': 3,
    'Class_5': 4,
    'Class_6': 5,
    'Class_7': 6,
    'Class_8': 7,
    'Class_9': 8,
}
train['target'] = train['target'].map(label_map)

# In[ ]:


features = ['feature_{}'.format(x) for x in range(75)]
qt = train[features].quantile(np.arange(0, 1, 0.002))


def clip(df):
    df = df.copy()
    for feature in features:
        df[feature] = df[feature].clip(lower=0, upper=qt.loc[0.998][feature])
    return df


# In[ ]:


values = []
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, ]
for feature in features:
    grouped = clip(train).groupby(feature)
    for value, group in grouped:
        value = [feature, value]
        for label in labels:
            p = (group['target'] == label).mean()
            p = np.clip(p, 1e-06, 1 - 1e-06)
            value.append(np.log(p + 0.5))
            value.append(np.log(p / (1 - p)))
        values.append(value)
df_proba = pd.DataFrame(values,
                        columns=['feature', 'value',
                                 'Class_1_proba1',
                                 'Class_1_proba2',
                                 'Class_2_proba1',
                                 'Class_2_proba2',
                                 'Class_3_proba1',
                                 'Class_3_proba2',
                                 'Class_4_proba1',
                                 'Class_4_proba2',
                                 'Class_5_proba1',
                                 'Class_5_proba2',
                                 'Class_6_proba1',
                                 'Class_6_proba2',
                                 'Class_7_proba1',
                                 'Class_7_proba2',
                                 'Class_8_proba1',
                                 'Class_8_proba2',
                                 'Class_9_proba1',
                                 'Class_9_proba2',
                                 ])
proba_dict_1 = {}
proba_dict_2 = {}

for i in range(len(df_proba)):
    feature = df_proba.iloc[i]['feature']
    value = df_proba.iloc[i]['value']
    proba_dict_1[feature, value] = df_proba.iloc[i][
        ['Class_1_proba1', 'Class_2_proba1', 'Class_3_proba1', 'Class_4_proba1', 'Class_5_proba1', 'Class_6_proba1',
         'Class_7_proba1', 'Class_8_proba1', 'Class_9_proba1', ]].values.astype(float)
    proba_dict_2[feature, value] = df_proba.iloc[i][
        ['Class_1_proba2', 'Class_2_proba2', 'Class_3_proba2', 'Class_4_proba2', 'Class_5_proba2', 'Class_6_proba2',
         'Class_7_proba2', 'Class_8_proba2', 'Class_9_proba2', ]].values.astype(float)

# In[ ]:


from sklearn.base import TransformerMixin


def reshape(df):
    values = []
    for value in df.values:
        values.append([_ for _ in value])
    return np.array(values)


class MyTransformer1(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        newX = pd.DataFrame()
        for feature in features:
            newX[feature] = X[feature].clip(lower=qt.loc[0.002][feature], upper=qt.loc[0.998][feature])
            newX[feature] = 1 / (newX[feature] - newX[feature].min() + 1)
        return newX


class MyTransformer2(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        newX = pd.DataFrame()
        X = clip(X)
        for feature in features:
            newX[feature] = X[feature].apply(lambda x: proba_dict_1[feature, x])
        return reshape(newX).reshape((-1, 75 * 9))


class MyTransformer3(MyTransformer2):
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        newX = pd.DataFrame()
        X = clip(X)
        for feature in features:
            newX[feature] = X[feature].apply(lambda x: proba_dict_1[feature, x])
        return reshape(newX)


def normalize(df, columns):
    """
    sklearn.preprocessing.MinMaxScaler
    """
    for column in columns:
        min_val, max_val = df[column].agg([min, max])
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df


class MyTransformer4(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        newX = pd.DataFrame()
        for feature in features:
            newX[feature] = X[feature].clip(lower=0).apply(lambda x: 1 / (x + 1))
        return normalize(newX, features)


class MyTransformer5(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        newX = pd.DataFrame()
        for feature in features:
            newX[feature] = X[feature].clip(lower=0).apply(lambda x: 1 / (x + 1))
        return normalize(newX, features)


class MyTransformer6(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        newX = pd.DataFrame()
        for feature in features:
            newX[feature] = X[feature].clip(lower=qt.loc[0.002][feature], upper=qt.loc[0.998][feature])
        return newX


# In[ ]:


from sklearn.base import ClassifierMixin
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# get_ipython().system('pip install lightautoml')
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from sklearn.pipeline import make_pipeline

my_model1 = CatBoostClassifier(
    iterations=890,
    min_child_samples=203,
    eval_metric='MultiClass',
    random_state=SEED,
    max_depth=1,
    verbose=True)

my_model2 = CatBoostClassifier(
    iterations=160,
    min_child_samples=30,
    max_depth=3,
    eval_metric='MultiClass',
    random_state=SEED,
    verbose=True)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam'
)


class TensorflowClassifier(ClassifierMixin):
    def __init__(self):
        self.histories = []
        self.classes_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(75, 9)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_crossentropy', ])

    def get_params(self, deep):
        return {}

    def fit(self, X, y):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = self.model.fit(X, y, epochs=300, batch_size=75 * 9, validation_split=0.1, callbacks=[callback],
                                 verbose=0)
        self.histories.append(history)
        return self

    def predict_proba(self, X):
        return self.model.predict(X).reshape((-1, 9))


my_model3 = TensorflowClassifier()

my_model4 = LGBMClassifier(
    random_state=SEED,
    min_child_samples=150,
    n_estimators=400,
    max_depth=3,
    reg_alpha=0.91,
    min_child_weight=0.001,
)

my_model5 = GradientBoostingClassifier(
    random_state=SEED,
    min_samples_leaf=36,
    n_estimators=100,

)


class AutoMLClassifier(ClassifierMixin):
    def __init__(self):
        task = Task('multiclass', metric='crossentropy', )
        self.model = TabularAutoML(
            task=task,
            timeout=900,
            general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]},
            reader_params={'cv': 20, 'random_state': SEED},
            tuning_params={'max_tuning_iter': 100, 'max_tuning_time': 100},
            lgb_params={'default_params': {'num_threads': 8}}, verbose=1)
        self.classes_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, ]
        self.train_prediction = None

    def get_params(self, deep):
        return {}

    def dataframe(self, X):
        if not isinstance(X, type(pd.DataFrame())):
            X = pd.DataFrame(X, columns=['column_{}'.format(x) for x in range(X.shape[1])])
        return X

    def fit(self, X, y):
        df = self.dataframe(X.copy())
        df['target'] = y
        self.train_prediction = self.model.fit_predict(df, roles={'target': 'target'}).data
        return self

    def predict_proba(self, X):
        X = self.dataframe(X)
        return self.model.predict(X).data


pipeline1 = make_pipeline(MyTransformer1(), my_model1)
pipeline2 = make_pipeline(MyTransformer2(), my_model2)
pipeline3 = make_pipeline(MyTransformer3(), my_model3)
pipeline4 = make_pipeline(MyTransformer4(), my_model4)
pipeline5 = make_pipeline(MyTransformer5(), my_model5)

# In[ ]:


my_final_estimator = AutoMLClassifier()

# In[ ]:


from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

voting_estimators = [
    ('mod1', pipeline1),
    ('mod2', pipeline2),
    ('mod3', pipeline3),
    ('mod4', pipeline4),
    ('mod5', pipeline5),
]
stacking_estimators = [
    ('mod1', pipeline1),
    ('mod2', pipeline2),
    ('mod3', pipeline3),
    ('mod4', pipeline4),
    ('mod5', pipeline5),
]

X = train[features]
y = train['target']

mod_vot = VotingClassifier(
    estimators=voting_estimators,
    voting='soft',
).fit(X, y)

mod_stk = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=my_final_estimator,
    stack_method='predict_proba',
    cv=20,
).fit(X, y)

# In[ ]:


y_pred_test = (mod_vot.predict_proba(test[features]) + mod_stk.predict_proba(test[features])) / 2
submission = test[['id']].copy()
submission['Class_1'] = y_pred_test[:, 0]
submission['Class_2'] = y_pred_test[:, 1]
submission['Class_3'] = y_pred_test[:, 2]
submission['Class_4'] = y_pred_test[:, 3]
submission['Class_5'] = y_pred_test[:, 4]
submission['Class_6'] = y_pred_test[:, 5]
submission['Class_7'] = y_pred_test[:, 6]
submission['Class_8'] = y_pred_test[:, 7]
submission['Class_9'] = y_pred_test[:, 8]
submission.to_csv('submission.csv', index=False)

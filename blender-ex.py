import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from BlenderN import Blender
from StackedGeneralizerN import StackedGeneralizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


### Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:,1:94].values
y_train = train.iloc[:,94].values
y_train = LabelEncoder().fit(y_train).transform(y_train)
X_test = test.iloc[:,1:94].values
y_train_pd = pd.get_dummies(y_train).as_matrix()


### Initialize Models
# Random Forest
RFA = RandomForestClassifier(n_estimators=100, n_jobs=14)
RFB = RandomForestClassifier(n_estimators=500, n_jobs=14)
RFC = RandomForestClassifier(n_estimators=1500, n_jobs=14)

# XGBoost
XGBA = xgb.XGBClassifier(objective='multi:softprob', nthread = 14, n_estimators=200, max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, min_child_weight=0.8, gamma=1)
XGBB = xgb.XGBClassifier(objective='multi:softprob', nthread = 14, n_estimators=400, max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, min_child_weight=0.8, gamma=1)
XGBC = xgb.XGBClassifier(objective='multi:softprob', nthread = 14, n_estimators=600, max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, min_child_weight=0.8, gamma=1)

# K-Nearest Neighbors
KNNA = KNeighborsClassifier(n_neighbors=10, n_jobs=14)
KNNB = KNeighborsClassifier(n_neighbors=50, n_jobs=14)
KNNC = KNeighborsClassifier(n_neighbors=100, n_jobs=14)

# Neural Networks
NNA = Sequential()
NNA.add(Dense(93, input_shape=(93,)))
NNA.add(Activation('relu'))
NNA.add(Dropout(.5))
NNA.add(Dense(50))
NNA.add(Activation('relu'))
NNA.add(Dropout(.5))
NNA.add(Dense(9))
NNA.add(Activation('softmax'))

NNB = Sequential()
NNB.add(Dense(93, input_shape=(93,)))
NNB.add(Activation('relu'))
NNB.add(Dropout(.5))
NNB.add(Dense(100))
NNB.add(Activation('relu'))
NNB.add(Dropout(.5))
NNB.add(Dense(9))
NNB.add(Activation('softmax'))

NNC = Sequential()
NNC.add(Dense(93, input_shape=(93,)))
NNC.add(Activation('relu'))
NNC.add(Dropout(.5))
NNC.add(Dense(200))
NNC.add(Activation('relu'))
NNC.add(Dropout(.5))
NNC.add(Dense(9))
NNC.add(Activation('softmax'))

NNA.compile(optimizer='sgd', loss='categorical_crossentropy')
NNB.compile(optimizer='sgd', loss='categorical_crossentropy')
NNC.compile(optimizer='sgd', loss='categorical_crossentropy')


### Initialize Blender
clfs = [RFA, RFB, RFC, XGBA, XGBB, XGBC, KNNA, KNNB, KNNC, NNA, NNB, NNC]
XGBlend = xgb.XGBClassifier(objective='multi:softprob', nthread = 14, n_estimators=435, max_depth=4, learning_rate=0.1, subsample=0.85, colsample_bytree=0.85, min_child_weight=0.8, gamma=5)
blender = Blender(clfs, ensemble = XGBlend, stratify=True, original = True, epoch=2000, batch_size=2000)
blender.fit(X_train, y_train)
pd.DataFrame(blender.predict(X_test)).to_csv('BDpredictions.csv')



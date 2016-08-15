/*
*
* Tag: Digit Recogintion
*/
import h2o
import time
import math
import datetime
import pandas as pd
import numpy as np
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

h2o.init()

from h2o.utils.shared_utils import _locate

train_file = [_locate("train.csv")]
test_file = [_locate("test.csv")]

print("Import and Parse train and test data")

train_all_data = h2o.import_file(path=train_file)
test_data = h2o.import_file(path=test_file)

#train_all_data.describe()
#test_data.describe()

#print(train_all_data.names)
#print(test_data.names)

global glm
digitrecog_x = train_all_data.names[1:]
digitrecog_y = train_all_data.names[0:1]
#print(digitrecog_x)
#print(digitrecog_y)
start_time = time.time()
glm = H2OGeneralizedLinearEstimator(alpha=[0.000001], Lambda=[0.001], family="gaussian", missing_values_handling="Skip", nfolds=10)
#glm.train(x=digitrecog_x, y=digitrecog_y, training_frame=train, validation_frame=valid)
glm.train(x=digitrecog_x, y=digitrecog_y, training_frame=train_all_data)
end_time = time.time() - start_time

train_glm = glm.model_performance(train_all_data).r2()
#test_glm = glm.model_performance(test_data).r2()

header = ["Model", "R2 TRAIN"]
table = [["GLM", train_glm]]
h2o.display.H2ODisplay(table, header)

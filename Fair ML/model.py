
from datetime import datetime

import numpy as np

from sklearn import svm

from Preprocessing import preprocess
from Report_Results import report_results
from utils import *
from Postprocessing import *

start_time = datetime.now()

metrics = ["sex", "age_cat", 'race', 'c_charge_degree', 'priors_count']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

np.random.seed(42)
SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000)
SVR.fit(training_data, training_labels)

training_class_predictions = SVR.predict(training_data)
training_prediction = list()
test_class_prediction = SVR.predict(test_data)
test_prediction = list()

for i in range(len(training_labels)):
    training_prediction.append(training_class_predictions[i])

for i in range(len(test_labels)):
    test_prediction.append(test_class_prediction[i])

training_race_cases = get_cases_by_metric(training_data, categories, 'race', mappings, training_prediction, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, 'race', mappings, test_prediction, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, epsilon = 0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

print('********** TRAIN DATA **********')
print('Accuracy on training data:')
print(str(round(get_total_accuracy(training_race_cases)*100, 2))+'%')
print('')

train_cost = apply_financials(training_race_cases)
print('Cost calculated for training data')
print('${:,.0f}'.format(train_cost))
print('')

print('********** TEST DATA **********')
print('Thresholds for each group')
for each_group in training_race_cases.keys():
    print('Threshold for '+each_group+' is '+str(thresholds[each_group]))
print('')

print('Accuracy for each group')
for each_group in training_race_cases.keys():
    print('Accuracy for '+each_group+' is '+str(round(get_num_correct(test_race_cases[each_group])/len(test_race_cases[each_group])*100, 2))+'%')
print('')

print('True positive rate (TPR) for each category')
for each_group in training_race_cases.keys():
    print('TPR of '+each_group+' is '+str(get_true_positive_rate(test_race_cases[each_group])))
print('')

print('Accuracy on test data:')
print(str(round(get_total_accuracy(test_race_cases)*100, 2))+'%')
print('')

test_cost = apply_financials(test_race_cases)
print('Cost calculated for test data')
print('${:,.0f}'.format(test_cost))
print('')

print('The cost for training data is ${:,.0f}'.format(train_cost)+' and for testing data, it is ${:,.0f}'.format(test_cost))
print("Therefore, the total cost is ${:,.0f}".format(train_cost + test_cost))
print('')

end_time = datetime.now()
print('Time taken: '+str(end_time - start_time)+' seconds')

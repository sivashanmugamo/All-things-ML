
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: COST
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""

import numpy
from utils import *

def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}

    # Must complete this function!

    categorical_predi = dict()
    categorical_ratio = dict()
    categorical_thres = dict()
    categorical_finan = dict()

    for each_category in categorical_results:
        categorical_predi[each_category] = list()
        categorical_ratio[each_category] = list()
        categorical_thres[each_category] = list()
        categorical_finan[each_category] = list()

        init_threshold = 0.01

        while init_threshold <= 1:

            temp_predi = apply_threshold(categorical_results[each_category], init_threshold)
            categorical_predi[each_category].append(temp_predi)

            temp_ratio = get_num_predicted_positives(temp_predi)/(len(categorical_results[each_category]))
            categorical_ratio[each_category].append(temp_ratio)

            temp_finan = apply_financials(temp_predi, True)
            categorical_finan[each_category].append(temp_finan)

            categorical_thres[each_category].append(init_threshold)

            init_threshold += 0.01

    min_money = numpy.NINF
    temp_list = list()
    temp_thres = list()

    for i in range(len(categorical_ratio['African-American'])):
        african_american_ratio = categorical_ratio['African-American'][i]

        for j in range(len(categorical_ratio['Caucasian'])):
            caucasian_ratio = categorical_ratio['Caucasian'][j]

            if(abs(african_american_ratio - caucasian_ratio) <= epsilon):
                for k in range(len(categorical_ratio['Hispanic'])):
                    hispanic_ratio = categorical_ratio['Hispanic'][k]

                    if(abs(caucasian_ratio - hispanic_ratio) <= epsilon) and (abs(african_american_ratio - hispanic_ratio) <=epsilon):
                        for l in range(len(categorical_ratio['Other'])):
                            other_ratio = categorical_ratio['Other'][l]

                            if(abs(hispanic_ratio - other_ratio) <= epsilon) and (abs(caucasian_ratio - other_ratio) <= epsilon) and (abs(african_american_ratio - other_ratio) <=epsilon):    
                                if (categorical_thres['African-American'][i], categorical_thres['Caucasian'][j], categorical_thres['Hispanic'][k], categorical_thres['Other'][l]) not in temp_thres:
                                    temp_thres.append((categorical_thres['African-American'][i], categorical_thres['Caucasian'][j], categorical_thres['Hispanic'][k], categorical_thres['Other'][l]))
                                    temp_money = categorical_finan['African-American'][i] + categorical_finan['Caucasian'][j] + categorical_finan['Hispanic'][k] + categorical_finan['Other'][l]
                                    temp_list.append(temp_money)

                                    if temp_money > min_money:
                                        min_money = temp_money
                                        best_threshold = (categorical_thres['African-American'][i], categorical_thres['Caucasian'][j], categorical_thres['Hispanic'][k], categorical_thres['Other'][l])

                                        demographic_parity_data['African-American'] = categorical_predi['African-American'][i]
                                        demographic_parity_data['Caucasian'] = categorical_predi['Caucasian'][j]
                                        demographic_parity_data['Hispanic'] = categorical_predi['Hispanic'][k]
                                        demographic_parity_data['Other'] = categorical_predi['Other'][l]

                                        thresholds['African-American'] = categorical_thres['African-American'][i]
                                        thresholds['Caucasian'] = categorical_thres['Caucasian'][j]
                                        thresholds['Hispanic'] = categorical_thres['Hispanic'][k]
                                        thresholds['Other'] = categorical_thres['Other'][l]

    return demographic_parity_data, thresholds

    return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""

def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}

    # Must complete this function!

    categorical_rate = dict()
    categorical_pred = dict()
    categorical_thre = dict()
    categorical_fina = dict()

    for each_category in categorical_results:
        categorical_rate[each_category] = list()
        categorical_pred[each_category] = list()
        categorical_thre[each_category] = list()
        categorical_fina[each_category] = list()

        init_threshold = 0.01

        while init_threshold <= 1:
            temp_pred = apply_threshold(categorical_results[each_category], init_threshold)
            categorical_pred[each_category].append(temp_pred)

            temp_rate = get_true_positive_rate(temp_pred)
            categorical_rate[each_category].append(temp_rate)

            temp_fina = apply_financials(temp_pred, True)
            categorical_fina[each_category].append(temp_fina)

            categorical_thre[each_category].append(init_threshold)

            init_threshold += 0.01

    min_money = numpy.NINF
    temp_list = list()
    temp_thre = list()

    for i in range(len(categorical_rate['African-American'])):
        african_american_rate = categorical_rate['African-American'][i]

        for j in range(len(categorical_rate['Caucasian'])):
            caucasian_rate = categorical_rate['Caucasian'][j]

            if(abs(caucasian_rate - african_american_rate) <= epsilon):
                for k in range(len(categorical_rate['Hispanic'])):
                    hispanic_rate = categorical_rate['Hispanic'][k]

                    if((abs(hispanic_rate - caucasian_rate) <= epsilon) and (abs(hispanic_rate - african_american_rate) <= epsilon)):
                        for l in range(len(categorical_rate['Other'])):
                            other_rate = categorical_rate['Other'][l]

                            if((abs(other_rate - hispanic_rate) <= epsilon) and (abs(other_rate - african_american_rate) <= epsilon) and (abs(other_rate - caucasian_rate) <= epsilon)):
                                threshold_combination = categorical_thre['African-American'][i], categorical_thre['Caucasian'][j], categorical_thre['Hispanic'][k], categorical_thre['Other'][l]

                                if threshold_combination not in temp_thre:
                                    temp_thre.append(threshold_combination)

                                    temp_money = categorical_fina['African-American'][i] + categorical_fina['Caucasian'][j] + categorical_fina['Hispanic'][k] + categorical_fina['Other'][l]
                                    temp_list.append(temp_money)

                                    if temp_money > min_money:
                                        min_money = temp_money
                                        best_threshold = threshold_combination

                                        equal_opportunity_data['African-American'] = categorical_pred['African-American'][i]
                                        equal_opportunity_data['Caucasian'] = categorical_pred['Caucasian'][j]
                                        equal_opportunity_data['Hispanic'] = categorical_pred['Hispanic'][k]
                                        equal_opportunity_data['Other'] = categorical_pred['Other'][l]

                                        thresholds['African-American'] = categorical_thre['African-American'][i]
                                        thresholds['Caucasian'] = categorical_thre['Caucasian'][j]
                                        thresholds['Hispanic'] = categorical_thre['Hispanic'][k]
                                        thresholds['Other'] = categorical_thre['Other'][l]

    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}

    # Must complete this function!

    categorical_PPV = {}
    mostMoneyCate = [4]
    bestThreshold = [4]
    mostMoney = numpy.NINF
    mIndex = 0
    justChanged = False

    for each_category in categorical_results:
        categorical_PPV[each_category] = list()
        init_threshold = 0.0

        for each_iteration in range(100):
            mp_data[each_category] = apply_threshold(categorical_results[each_category], init_threshold)
            categorical_PPV[each_category].append(get_positive_predictive_value(mp_data[each_category]))
            init_threshold += 0.01

            if(apply_financials(mp_data) > mostMoney):
                thresholds = init_threshold
                mostMoney = apply_financials(mp_data)

                if(not justChanged):
                    mostMoneyCate.pop()
                    mostMoneyCate.append(mostMoney)
                    bestThreshold.pop()
                    bestThreshold.append(init_threshold-0.01)

                else:
                    #print("In Not Just Changed")
                    mostMoneyCate.append(mostMoney)
                    bestThreshold.append(init_threshold - 0.01)

                justChanged = False

        justChanged = True
        mIndex += 1
        mostMoney = numpy.NINF

    mp_data['African-American'] = apply_threshold(categorical_results['African-American'], bestThreshold[0])
    mp_data['Caucasian'] = apply_threshold(categorical_results['Caucasian'], bestThreshold[1])
    mp_data['Hispanic'] = apply_threshold(categorical_results['Hispanic'], bestThreshold[2])
    mp_data['Other'] = apply_threshold(categorical_results['Other'], bestThreshold[3])


    thresholds = {'African-American': bestThreshold[0], 'Caucasian': bestThreshold[1], 'Hispanic': bestThreshold[2], 'Other': bestThreshold[3]}

    return mp_data, thresholds

    return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    # Must complete this function!

    categorical_predi = dict()
    categorical_ratio = dict()
    categorical_thres = dict()
    categorical_finan = dict()

    for each_category in categorical_results:
        categorical_predi[each_category] = list()
        categorical_ratio[each_category] = list()
        categorical_thres[each_category] = list()
        categorical_finan[each_category] = list()

        init_threshold = 0.01

        while init_threshold <= 1:

            temp_predi = apply_threshold(categorical_results[each_category], init_threshold)
            categorical_predi[each_category].append(temp_predi)

            temp_ratio = get_positive_predictive_value(temp_predi)
            categorical_ratio[each_category].append(temp_ratio)

            temp_finan = apply_financials(temp_predi, True)
            categorical_finan[each_category].append(temp_finan)

            categorical_thres[each_category].append(init_threshold)

            init_threshold += 0.01

    
    min_money = numpy.NINF
    temp_list = list()
    temp_thres = list()

    for i in range(len(categorical_ratio['African-American'])):
        african_american_ratio = categorical_ratio['African-American'][i]

        for j in range(len(categorical_ratio['Caucasian'])):
            caucasian_ratio = categorical_ratio['Caucasian'][j]

            if(abs(african_american_ratio - caucasian_ratio) <= epsilon):
                for k in range(len(categorical_ratio['Hispanic'])):
                    hispanic_ratio = categorical_ratio['Hispanic'][k]

                    if(abs(caucasian_ratio - hispanic_ratio) <= epsilon) and (abs(african_american_ratio - hispanic_ratio) <=epsilon):
                        for l in range(len(categorical_ratio['Other'])):
                            other_ratio = categorical_ratio['Other'][l]

                            if(abs(hispanic_ratio - other_ratio) <= epsilon) and (abs(caucasian_ratio - other_ratio) <= epsilon) and (abs(african_american_ratio - other_ratio) <=epsilon):
                                if (categorical_thres['African-American'][i], categorical_thres['Caucasian'][j], categorical_thres['Hispanic'][k], categorical_thres['Other'][l]) not in temp_thres:
                                    temp_thres.append((categorical_thres['African-American'][i], categorical_thres['Caucasian'][j], categorical_thres['Hispanic'][k], categorical_thres['Other'][l]))
                                    temp_money = categorical_finan['African-American'][i] + categorical_finan['Caucasian'][j] + categorical_finan['Hispanic'][k] + categorical_finan['Other'][l]
                                    temp_list.append(temp_money)

                                    if temp_money > min_money:
                                        min_money = temp_money
                                        best_threshold = (categorical_thres['African-American'][i], categorical_thres['Caucasian'][j], categorical_thres['Hispanic'][k], categorical_thres['Other'][l])

                                        predictive_parity_data['African-American'] = categorical_predi['African-American'][i]
                                        predictive_parity_data['Caucasian'] = categorical_predi['Caucasian'][j]
                                        predictive_parity_data['Hispanic'] = categorical_predi['Hispanic'][k]
                                        predictive_parity_data['Other'] = categorical_predi['Other'][l]

                                        thresholds['African-American'] = categorical_thres['African-American'][i]
                                        thresholds['Caucasian'] = categorical_thres['Caucasian'][j]
                                        thresholds['Hispanic'] = categorical_thres['Hispanic'][k]
                                        thresholds['Other'] = categorical_thres['Other'][l]

    return predictive_parity_data, thresholds

    return None, None

###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    # Must complete this function!

    init_threshold = 0.01

    threshold_applied_data = dict()

    min_money = numpy.NINF

    while init_threshold <= 1:

        threshold_applied_data['African-American'] = apply_threshold(categorical_results['African-American'], init_threshold)
        threshold_applied_data['Caucasian'] = apply_threshold(categorical_results['Caucasian'], init_threshold)
        threshold_applied_data['Hispanic'] = apply_threshold(categorical_results['Hispanic'], init_threshold)
        threshold_applied_data['Other'] = apply_threshold(categorical_results['Other'], init_threshold)

        temp_money = apply_financials(threshold_applied_data)

        if temp_money > min_money:
            min_money = temp_money

            single_threshold_data['African-American'] = apply_threshold(categorical_results['African-American'], init_threshold)
            single_threshold_data['Caucasian'] = apply_threshold(categorical_results['Caucasian'], init_threshold)
            single_threshold_data['Hispanic'] = apply_threshold(categorical_results['Hispanic'], init_threshold)
            single_threshold_data['Other'] = apply_threshold(categorical_results['Other'], init_threshold)

            thresholds['African-American'] = init_threshold
            thresholds['Caucasian'] = init_threshold
            thresholds['Hispanic'] = init_threshold
            thresholds['Other'] = init_threshold

        init_threshold += 0.01

    return single_threshold_data, thresholds

    return None, None

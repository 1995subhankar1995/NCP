
import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torchvision
import os
import pickle
import sys
import argparse
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import rankdata
from numpy.random import default_rng
from torch.nn.functional import softmax
from scipy.stats.mstats import mquantiles
import torch.nn as nn
import torch.optim as optim
import bisect


# The HPS non-conformity score
def class_probability_score(probabilities, labels, u=None, all_combinations=False):

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # calculate scores of each point with all labels
    if all_combinations:
        scores = 1 - probabilities[:, labels]

    # calculate scores of each point with only one label
    else:
        scores = 1 - probabilities[np.arange(num_of_points), labels]

    # return scores
    return scores


# The APS non-conformity score
def generalized_inverse_quantile_score(probabilities, labels, u=None, all_combinations=False):

    # whether to do a randomized score or not
    if u is None:
        randomized = False
    else:
        randomized = True

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # sort probabilities from high to low
    sorted_probabilities = -np.sort(-probabilities)

    # create matrix of cumulative sum of each row
    cumulative_sum = np.cumsum(sorted_probabilities, axis=1)

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[np.arange(num_of_points), labels] - 1

    # compute the scores of each label in each row
    scores = cumulative_sum[np.arange(num_of_points), label_ranks.T].T

    # compute the probability of the last label that enters
    last_label_prob = sorted_probabilities[np.arange(num_of_points), label_ranks.T].T

    # remove the last label probability or a multiplier of it in the randomized score
    if not randomized:
        scores = scores - last_label_prob
    else:
        scores = scores - np.diag(u) @ last_label_prob

    # return the scores
    return scores


#def APS_score(outputs, I, ordered, cumsum, np.arange(num_of_classes), uniform_variables, all_combinations=True):
    

# The RAPS non-conformity score
def rank_regularized_score(probabilities, labels, u=None, all_combinations=False, lamda = 0.0, k_reg = 5):

    # get the regular scores
    scores = generalized_inverse_quantile_score(probabilities, labels, u, all_combinations)

    # get number of classes
    num_of_classes = np.shape(probabilities)[1]

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[np.arange(num_of_points), labels] - 1

    #print(f"label_ranks = {label_ranks}")
    #exit(1)
    tmp = label_ranks+1-k_reg
    tmp[tmp < 0] = 0
    scores = scores + lamda * tmp
    scores[scores > 1] = 1

    # return scores
    return scores

def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        #print(f"T = {T.item()}")
        for x, targets in calib_loader:
            #print(x.shape)
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 


def get_logits_targets(model, loader, BatchSize = 1024, NumClass = 10, device = 'cpu'):
    logits = torch.zeros((len(loader.dataset), NumClass))
    labels = torch.zeros((len(loader.dataset),))

    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.cuda()).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    #dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return logits, labels.long()



class SplitCP():
    """
    SplitCP: This is the conformal prediction method that needs a trained model and some data for calibration. In practice, we randomly split data into two parts: training and calibration data. 
    We train a neural network model using the training data and find a threshold using calibration data. Later, we use the threshold to estimate a prediction set for a new data point.
    """
    def __init__(self, model:torch.nn.Module, ScoresList: list, alpha: float, NumExperiments: int, CoverageOnLabel: bool, SavePath: str, device: str, cal_test_ratio: float, BatchSize: int, PlatScale: bool):
        """
        args:
            model: A trained model that gives logits as outputs. The shape of the outputs from the model would be (num_samples, num_classes).
            
            ScoresList: A list of non-conformity scores. It will be either ['APS'] or ['HPS'] or ['APS', 'HPS']. 'APS' represents adaptive prediction 
            set given in "https://arxiv.org/abs/2006.02544" and 'HPS' represents homogeneous prediction set given in "https://www.stat.cmu.edu/~ryantibs/statml/lectures/Lei-Robins-Wasserman.pdf".

            cal_test_ratio: the ratio in range of (0, 1) in which we want to devide the data into two parts calibration set and test set.

            alpha: the miscoverage label.

            NumExperiments: For how many runs we need the results.

            CoverageOnLabel: This gives coverage for each class if we make it True.

            device: Specify either 'cuda:number' or 'cpu' depending on the configuration of the sytems. It is advisable to run on 'cuda'.

            BatchSize: How many samples per batch to load.

            PlatScale: Temperature scaling we need to apply before applying CP methods.

            SavePath: Specify the path we want to save the results.
         """
        self.model = model
        self.cal_test_ratio = cal_test_ratio
        self.ScoresList = ScoresList
        self.alpha = alpha
        self.NumExperiments = NumExperiments
        self.CoverageOnLabel = CoverageOnLabel
        self.SavePath = SavePath
        self.device = device
        self.BatchSize = BatchSize
        self.PlatScale = PlatScale
        assert len(self.ScoresList) <=2, 'There are only two base non-conformity scores for classification.'
        
    def get_scores(self, fx, y, indices, num_of_classes, T, lamda = None, k_reg = None, Vanilla = True):
        """
        We introduce lamda and k_reg for RAPS method only.
        
        """
        n = fx.size()[0]

        # create container for the scores
        if Vanilla:
            scores_simple = np.zeros((len(self.ScoresList), n, num_of_classes))
        else:
            scores_simple = np.zeros((1, n, num_of_classes))
        rng = default_rng()
        uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)
        simple_outputs = softmax(fx/T, dim=1).numpy()


        if Vanilla:
            for p, score in enumerate(self.ScoresList):
                if score == 'APS':
                    scores_simple[p, :, :] = generalized_inverse_quantile_score(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
                    #scores_simple[p, :, :] = APS_score(outputs, I, ordered, cumsum, np.arange(num_of_classes), uniform_variables, all_combinations=True)
                elif score == 'HPS':
                    scores_simple[p, :, :] = class_probability_score(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:    
            scores_simple[0, :, :] = rank_regularized_score(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True, lamda = lamda, k_reg = k_reg)


        return scores_simple

    def calibration(self, scores_simple=None, Vanilla = True):
        # size of the calibration set
        n_calib = scores_simple.shape[1]
        #print(f"n_calib = {n_calib}")

        # create container for the calibration thresholds

        # Compute thresholds
        level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(n_calib))
        if Vanilla:
            thresholds = {} #np.zeros((len(self.ScoresList), 1))
            for p in range(len(self.ScoresList)):
                thresholds[self.ScoresList[p]] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds = np.zeros((1, 1))
            thresholds[0, 0] = mquantiles(scores_simple[0, :], prob=level_adjusted)            

        return thresholds

    def prediction(self, SoftmaxScores = None, scores_simple=None, thresholds=None, Vanilla = True):
        n = scores_simple.shape[1]
        predicted_sets = []
        if Vanilla:
            for p, key in enumerate(thresholds):
                S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[key])[0].tolist() for i in range(n)]
                # No empty set is allowed
                for k in range(len(S_hat_simple)):
                    if len(S_hat_simple[k]) == 0:
                        S_hat_simple[k] = [np.argsort(scores_simple[p, k, :])[0]]

                predicted_sets.append(S_hat_simple)

        else:
            S_hat_simple = [np.where(scores_simple[0, i, :] <= thresholds[0, 0])[0] for i in range(n)]
            for k in range(len(S_hat_simple)):
                if len(S_hat_simple[k]) == 0:
                    S_hat_simple[k] = [np.argsort(scores_simple[0, k, :])[0]]
            predicted_sets.append(S_hat_simple)            
        return predicted_sets


    def evaluate_predictions(self, S, X, y, conditional=False, coverage_on_label=False, num_of_classes=10):

        for i in range(len(S)):
            if len(S[i]) == 0:
                print(S[i])

        marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

        # If desired calculate coverage for each class
        if coverage_on_label:
            sums = np.zeros(num_of_classes)
            size_sums = np.zeros(num_of_classes)
            lengths = np.zeros(num_of_classes)
            for i in range(len(y)):
                lengths[y[i]] = lengths[y[i]] + 1
                size_sums[y[i]] = size_sums[y[i]] + len(S[i])
                if y[i] in S[i]:
                    sums[y[i]] = sums[y[i]] + 1
            coverage_given_y = sums/lengths
            lengths_given_y = size_sums/lengths

        # Conditional coverage not implemented
        wsc_coverage = None

        # Size and size conditional on coverage
        size = np.mean([len(S[i]) for i in range(len(y))])
        idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
        size_cover = np.mean([len(S[i]) for i in idx_cover])

        # Combine results
        out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                            'Size': [size], 'Size cover': [size_cover]})

        # If desired, save coverage for each class
        if coverage_on_label:
            for i in range(num_of_classes):
                out['Coverage given '+str(i)] = coverage_given_y[i]
                out['Size given '+str(i)] = lengths_given_y[i]

        return out


    def VanillaCP(self, DataLoader, num_of_classes):
        # Find Logits
        fx_test, y_test = get_logits_targets(self.model, DataLoader, BatchSize = self.BatchSize, NumClass = num_of_classes, device = self.device)

        if self.PlatScale:
            n_test = len(y_test)
            indices = torch.arange(n_test)
            idx1, idx2 = train_test_split(indices, test_size=0.5, random_state = 2023)
            TempScaleTensorData = torch.utils.data.TensorDataset(fx_test[idx1], y_test[idx1].long())
            TempScaleLoader = torch.utils.data.DataLoader(TempScaleTensorData, batch_size = 128, shuffle = False, pin_memory = True)
            self.T = platt_logits(TempScaleLoader).item()
        else: 
            self.T = 1.0

        print(f"T Vanilla value = {self.T}")

        n_test = len(y_test)
        indices = torch.arange(n_test)
        # get base scores of whole clean test set
        scores_simple_clean_test = self.get_scores(fx_test, y_test, indices, num_of_classes, self.T, Vanilla = True)

        # create dataframe for storing results
        results1 = pd.DataFrame()

        # container for storing bounds on "CP+SS"
        quantiles = np.zeros((len(self.ScoresList), 2, self.NumExperiments))

        # run for n_experiments data splittings
        print("\nRunning experiments for "+str(self.NumExperiments)+" random splits:\n")
        for experiment in tqdm(range(self.NumExperiments)):

            results = pd.DataFrame()

            # Split test data into calibration and test
            idx1, idx2 = train_test_split(indices, test_size=self.cal_test_ratio, random_state = experiment + 2023)

            # calibrate base model with the desired scores and get the thresholds
            #print(y_test[idx1])
            thresholds = self.calibration(scores_simple=scores_simple_clean_test[:, idx1, y_test[idx1]])
            print(f"Vanilla CP thresholds = {thresholds}")
            
            predicted_clean_sets = self.prediction(scores_simple=scores_simple_clean_test[:, idx2, :], thresholds=thresholds)
            #print(li for li in predicted_clean_sets[1] if len(li) == 0)
            #exit(1)

            # arrange results on clean test set in dataframe
            for p in range(len(self.ScoresList)):
                res = self.evaluate_predictions(predicted_clean_sets[p], None, y_test[idx2].numpy(),
                                            conditional=False,coverage_on_label=False, num_of_classes=num_of_classes)
                #res = coverage_size(predicted_clean_sets[p],y_test[idx2].numpy())
                
                res['Score'] = self.ScoresList[p]
                results = results.append(res)
            results['Experiment'] = str(experiment)
            results1 = results1.append(results)
            del idx1, idx2, predicted_clean_sets, thresholds
            gc.collect()
        return results1


    def PickKreg(self, paramtune_logits, labels):
        paramtune_logits = paramtune_logits.numpy()
        labels = labels.numpy()
        #label_ranks = rankdata(-paramtune_logits, method='ordinal', axis=1)[np.arange(len(labels)), labels] - 1
        gt_locs_kstar = np.array([(np.where(np.argsort(-paramtune_logits[i]) == labels[i].item())[0]) for i in range(len(paramtune_logits))])
        kstar = np.quantile(gt_locs_kstar, 1-self.alpha, interpolation='higher') + 1
        return kstar 



    def RegularizeCP(self, DataLoader, num_of_classes, k_reg = None, lamda = None):
        fx_test, y_test = get_logits_targets(self.model, DataLoader, BatchSize = self.BatchSize, NumClass = num_of_classes, device = self.device)

        if self.PlatScale:
            n_test = len(y_test)
            indices = torch.arange(n_test)
            idx1, idx2 = train_test_split(indices, test_size=0.5, random_state = 2023)
            TempScaleTensorData = torch.utils.data.TensorDataset(fx_test[idx1], y_test[idx1].long())
            TempScaleLoader = torch.utils.data.DataLoader(TempScaleTensorData, batch_size = 128, shuffle = False, pin_memory = True)
            self.T = platt_logits(TempScaleLoader).item()
        else: 
            self.T = 1.0

        results1 = pd.DataFrame()
        self.num_of_classes = num_of_classes


        for experiment in tqdm(range(self.NumExperiments)):
            results = pd.DataFrame()
            if k_reg == None or lamda == None:
                n_test = len(y_test)
                indices = torch.arange(n_test)
                idx_cal, idx_test = train_test_split(indices, test_size=0.50, random_state = 2023 + experiment)
                idx_cal, idx_hyper = train_test_split(idx_cal, test_size=0.30, random_state = 2023 + experiment)


                if k_reg == None:
                    k_reg = self.PickKreg(fx_test[idx_hyper], y_test[idx_hyper])

                n_test = len(y_test)
                indices = torch.arange(n_test)
                best_size = self.num_of_classes
                for temp_lam in [0.001, 0.01, 0.15, 0.2, 0.5]:
                    scores_simple_clean_test = self.get_scores(fx_test, y_test, indices, num_of_classes, self.T, temp_lam, k_reg, Vanilla = False)

                    thresholds = self.calibration(scores_simple=scores_simple_clean_test[:, idx_cal, y_test[idx_cal]], Vanilla = False)

                    # generate prediction sets on the clean test set for base model
                    predicted_clean_sets = self.prediction(scores_simple=scores_simple_clean_test[:, idx_hyper, :], thresholds=thresholds, Vanilla = False)
                    #print(predicted_clean_sets)
                    res = self.evaluate_predictions(predicted_clean_sets[0], None, y_test[idx_hyper].numpy(),
                                                conditional=False,coverage_on_label=False, num_of_classes=num_of_classes)

                    sz_avg = res['Size'].to_numpy()[0]
                    if sz_avg < best_size:
                        best_size = sz_avg
                        lamda_star = temp_lam
            scores_simple_clean_test = self.get_scores(fx_test, y_test, indices, num_of_classes, self.T, lamda_star, k_reg, Vanilla = False)

            thresholds = self.calibration(scores_simple=scores_simple_clean_test[:, idx_cal, y_test[idx_cal]], Vanilla = False)
            print(f"RCP thresholds = {thresholds}")

            # generate prediction sets on the clean test set for base model
            predicted_clean_sets = self.prediction(scores_simple=scores_simple_clean_test[:, idx_test, :], thresholds=thresholds, Vanilla = False)
            res = self.evaluate_predictions(predicted_clean_sets[0], None, y_test[idx_test].numpy(),
                                        conditional=False,coverage_on_label=False, num_of_classes=num_of_classes)                





            res['Score'] = 'RAPS'
            results = results.append(res)
            results['Experiment'] = str(experiment)
            results1 = results1.append(results)
            del idx_cal, idx_hyper, idx_test, predicted_clean_sets, thresholds
            gc.collect()
        return results1


    def FindThresholds(self, scores_simple=None, Vanilla = True):
        # size of the calibration set
        n_calib = scores_simple.shape[1]
        #print(f"n_calib = {n_calib}")

        # create container for the calibration thresholds

        #print(f"scores_simple shape1 = {scores_simple.shape}")
        # Compute thresholds
        level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(n_calib))
        if Vanilla:
            thresholds = np.zeros((len(self.ScoresList), scores_simple.shape[1])) #np.zeros((len(self.ScoresList), 1))
            for p in range(len(self.ScoresList)):
                for j in range(scores_simple.shape[1]):
                    thresholds[p, j] = mquantiles(scores_simple[p, j, :], prob=level_adjusted)
        else:
            thresholds = np.zeros((1, 1))
            thresholds[0, 0] = mquantiles(scores_simple[0, :], prob=level_adjusted)            

        return thresholds

        def prediction_ProbabilisticLCP(self, scores_simple=None, thresholds=None, Regularization = False):
        n = scores_simple.shape[1]
        predicted_sets = []

        if Regularization:
            ScoresList = ['APS']
        else:
            ScoresList = self.ScoresList

        for p in range(len(ScoresList)):
            S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, i])[0].tolist() for i in range(n)]
            # No empty set is allowed
            for k in range(len(S_hat_simple)):
                if len(S_hat_simple[k]) == 0:
                    S_hat_simple[k] = [np.argsort(scores_simple[p, k, :])[0]]
            predicted_sets.append(S_hat_simple)
    
        return predicted_sets

    def HypertuningBothLamdas(self, fx_test, idx_hyper, idx_cal, scores_simple_clean_test, y_test, ):

        best_size = self.num_of_classes
        k_values = len(idx_cal)

        assert len(idx_cal) > 500
        KNNs = np.linspace(500, len(idx_cal), 3)
        KNNs = [int(ele) for ele in KNNs]

        for knn in KNNs:
            max_lamda = 1e5
            min_lamda = 1

            while max_lamda - min_lamda > 1:
                local_lamda = (max_lamda+min_lamda)/2.0
                distance_mat = torch.exp(-(torch.cdist(fx_test[idx_hyper], fx_test[idx_cal], p=2)) / local_lamda)



                cal_score = torch.from_numpy(scores_simple_clean_test[:, idx_cal, y_test[idx_cal]]).unsqueeze(dim = 1).repeat(1, len(idx_hyper), 1).numpy()*distance_mat.numpy()

                cal_score = torch.from_numpy(cal_score)

                sorted_mat, indices = torch.sort(-1 * distance_mat, dim = 1)

                k_indices = indices[:, :knn]

                #print('ha')
                cal_score_knns = torch.tensor([cal_score.squeeze(dim = 0)[i][k_indices[i]].numpy().tolist() for i in range(len(k_indices))])
                #print('na')
                cal_score_knns = cal_score_knns.unsqueeze(dim = 0).numpy()

                thresholds = self.FindThresholds(cal_score_knns)
                #print(f"thresholds = {thresholds}")



                predicted_clean_sets = self.prediction_ProbabilisticLCP(scores_simple=scores_simple_clean_test[:, idx_hyper, :], thresholds=thresholds, Regularization = True)


                res = self.evaluate_predictions(predicted_clean_sets[0], None, y_test[idx_hyper].numpy(),
                                        conditional=False,coverage_on_label=False, num_of_classes=self.num_of_classes)

                sz_avg = res['Size'].to_numpy()[0]
                cvg_avg = res['Coverage'].to_numpy()[0]

                #sz_avg, cvg_avg = self.HyperTuning()

                #print(f" size = {sz_avg} cvg = {cvg_avg} local value min = {min_lamda} local value max = {max_lamda} local_lamda = {local_lamda} knn = {knn}")
                if abs(cvg_avg - (1 - self.alpha)) < 0.01:
                    if sz_avg < best_size:
                        best_size = sz_avg
                        lamda_local_star = local_lamda
                        k_values = knn

                        break
                    #if abs(cvg_avg - (1 - self.alpha)) <= 0.01:
                    #    break

                if cvg_avg >= 1 - self.alpha:
                    max_lamda = local_lamda
                else:
                    min_lamda = local_lamda
        return lamda_local_star, best_size, k_values

    def TestProcedure(self, fx_test, y_test, idx_test, idx_cal, k_star = 0.0, lamda_local_star = torch.tensor([1]), k_reg = 0.0, lamda_star = 0.0, experiment = int(1)):
        results = pd.DataFrame()

        n_test = len(y_test)
        indices = torch.arange(n_test)
        scores_simple_clean_test = self.get_scores(fx_test, y_test, indices, self.num_of_classes, self.T, lamda_star, k_reg, Vanilla = False)

        distance_mat = torch.exp(-(torch.cdist(fx_test[idx_test], fx_test[idx_cal], p=2)) / lamda_local_star)

        #distance_mat = distance_mat.numpy()
        cal_score = torch.from_numpy(scores_simple_clean_test[:, idx_cal, y_test[idx_cal]]).unsqueeze(dim = 1).repeat(1, len(idx_test), 1).numpy()*distance_mat.numpy()

        cal_score = torch.from_numpy(cal_score)

        sorted_mat, indices = torch.sort(-1 * distance_mat)
        k_indices = indices[:, :k_star]

        cal_score_knns = torch.tensor([cal_score.squeeze(dim = 0)[i][k_indices[i]].numpy().tolist() for i in range(len(k_indices))])
        cal_score_knns = cal_score_knns.unsqueeze(dim = 0).numpy()

        
        thresholds = self.FindThresholds(cal_score_knns)



        predicted_clean_sets = self.prediction_ProbabilisticLCP(scores_simple=scores_simple_clean_test[:, idx_test, :], thresholds=thresholds, Regularization = True)

        res = self.evaluate_predictions(predicted_clean_sets[0], None, y_test[idx_test].numpy(),
                                    conditional=False,coverage_on_label=False, num_of_classes=self.num_of_classes)                





        res['Score'] = 'RAPS'
        results = results.append(res)
        results['Experiment'] = str(experiment)

        return results


    def DistanceLCP(self, DataLoader, num_of_classes, k_reg = None, lamda = None):
        results_R_LCP = pd.DataFrame()
        results_LCP = pd.DataFrame()


        fx_test, y_test = get_logits_targets(self.model, DataLoader, BatchSize = self.BatchSize, NumClass = num_of_classes, device = self.device)

        if self.PlatScale:
            n_test = len(y_test)
            indices = torch.arange(n_test)
            idx1, idx2 = train_test_split(indices, test_size=0.5, random_state = 2023)
            TempScaleTensorData = torch.utils.data.TensorDataset(fx_test[idx1], y_test[idx1].long())
            TempScaleLoader = torch.utils.data.DataLoader(TempScaleTensorData, batch_size = 128, shuffle = False, pin_memory = True)
            self.T = platt_logits(TempScaleLoader).item()
        else: 
            self.T = 1.0

        self.num_of_classes = num_of_classes

        for experiment in tqdm(range(self.NumExperiments)):
            if k_reg == None or lamda == None:
                n_test = len(y_test)
                indices = torch.arange(n_test)
                idx_cal, idx_test = train_test_split(indices, test_size=0.50, random_state = 2023 + experiment)
                idx_cal, idx_hyper = train_test_split(idx_cal, test_size=0.30, random_state = 2023 + experiment)


                if k_reg == None:
                    k_reg = self.PickKreg(fx_test[idx_hyper], y_test[idx_hyper])

                n_test = len(y_test)
                indices = torch.arange(n_test)
                lamda_local_star = 1e10
                best_size = self.num_of_classes

                for temp_lam in [0.0, 0.001, 0.01, 0.1, 0.2, 0.5]:
                    scores_simple_clean_test = self.get_scores(fx_test, y_test, indices, num_of_classes, self.T, temp_lam, k_reg, Vanilla = False)

                    if temp_lam == 0.0:
                        lamda_local_star_base, _, k_star_base = self.HypertuningBothLamdas(fx_test, idx_hyper, idx_cal, scores_simple_clean_test, y_test)
                    else:
                        lamda_local, lamda_size, k_values = self.HypertuningBothLamdas(fx_test, idx_hyper, idx_cal, scores_simple_clean_test, y_test)
                        if lamda_size < best_size:
                            best_size = lamda_size
                            lamda_star = temp_lam
                            lamda_local_star = lamda_local
                            k_star = k_values


            #print(f"lamda_star = {lamda_star}, lamda_local_star = {lamda_local_star}")

            #Results R-Distanced LCP
            results_r_lcp = self.TestProcedure(fx_test, y_test, idx_test, idx_cal, k_star = k_star, lamda_local_star = lamda_local_star, k_reg = k_reg, lamda_star = lamda_star, experiment = experiment)
            results_R_LCP = results_R_LCP.append(results_r_lcp)
            
            #Results Distanced LCP
            results_lcp = self.TestProcedure(fx_test, y_test, idx_test, idx_cal, k_star = k_star_base, lamda_local_star = lamda_local_star_base, k_reg = 0.0, lamda_star = 0.0, experiment = experiment)
            results_LCP = results_LCP.append(results_lcp)

        return results_R_LCP, results_LCP

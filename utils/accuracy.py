from sklearn import metrics
import os

class Accuracy:
    acc = 0
    rate_overkill = 0 # Pass but predict fail
    rate_underkill = 0 # Importance fail but predict pass
    unknow = 0
    nClassPass = 0
    nClassFail = 0
    nOverkill_fail = 0
    nUnderkill_fail = 0
    UK_list = []
    UK_list_predict = []
    UK_score = []
    UK_true = []
    OK_list = []
    OK_list_predict = []
    OK_score = []
    OK_true = []
    


    def __init__(self) -> None:
        self.acc = 0
        self.rate_overkill = 0 # Pass but predict fail
        self.rate_underkill = 0 # Importance fail but predict pass
        self.unknow = 0
        self.nClassPass = 0
        self.nClassFail = 0
        self.nOverkill_fail = 0
        self.nUnderkill_fail = 0
        self.UK_list = []
        self.UK_list_predict = []
        self.UK_score = []
        self.UK_true = []
        self.OK_list = []
        self.OK_list_predict = []
        self.OK_score = []
        self.OK_true = []

    def accuracy_score(self,truth, predict):
        self.acc  = metrics.accuracy_score(truth, predict)
        return self.acc

    def get_accuracy(self):
        return self.acc

    def Overkill_Underkill_score(self, truth, predict, pass_class, scores):
        '''pass_class must to encoding to interger list. The last one is a fail class. Reruen overkill, underkill'''
        nClassPass = 0
        nClassFail = 0
        nOverkill_fail = 0
        nUnderkill_fail = 0
        unknow = 0
        for i, (y_true, y_pred,score) in enumerate(zip(truth, predict,scores)):
            if y_true in pass_class: #true pass class
                nClassPass+=1
                if y_true != y_pred:
                    if y_pred not in pass_class:#pred fail class
                        self.OK_list.append(i)
                        self.OK_list_predict.append(y_pred)
                        self.OK_score.append(score)
                        self.OK_true.append(y_true)
                        nOverkill_fail+=1
                    else: #same pass but wrong class
                        unknow+=1

            else:
                nClassFail+=1
                if y_true != y_pred:

                    if y_pred in pass_class: #pred pass class
                        self.UK_list.append(i)
                        self.UK_list_predict.append(y_pred)
                        self.UK_score.append(score)
                        self.UK_true.append(y_true)
                        nUnderkill_fail +=1
                    else: # same fail but wrong class 
                        unknow+=1

        self.rate_overkill = nOverkill_fail * 1. / nClassPass
        self.rate_underkill = nUnderkill_fail * 1. / nClassFail
        self.unknow = unknow
        self.nClassFail = nClassFail
        self.nClassPass = nClassPass
        return [self.rate_overkill, self.rate_underkill]

    def get_unknow(self):
        return self.unknow

    def getUnderkill(self):
        return self.UK_list

    def getOverkill(self):
        return self.OK_list

        

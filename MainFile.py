#======================= IMPORT PACKAGES =============================

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#===================== DATA SELECTION ==============================

#=== READ A DATASET ====

print("===========================================")
print("------------ Data Selection ---------------")
print("===========================================")
print()
data_frame=pd.read_csv("kddcup99.csv")
data_frame=data_frame[0:200000]
print(data_frame.head(20))


#================= PREPROCESSING ======================================

#==== CHECKING MISSING VALUES =====

print("=============================================")
print("-------- Checking Missing values ------------")
print("=============================================")
print()
print(data_frame.isnull().sum())
print()

#==== LABEL ENCODING  ======

print("=============================================")
print("--------- Before Label Encoding -------------")
print("=============================================")
print()
print(data_frame.head(10))

label_encoder = preprocessing.LabelEncoder() 
print("=============================================")
print("---------- After Label Encoding -------------")
print("=============================================")
print()
label_encoder = preprocessing.LabelEncoder() 
data_frame['label']= label_encoder.fit_transform(data_frame['label']) 
data_frame['label'].unique() 
categ = ['protocol_type','service','flag']
label_encoder = preprocessing.LabelEncoder()
data_frame[categ] = data_frame[categ].apply(label_encoder.fit_transform)
print(data_frame.head(10))

#========================= DATA NORMALIZATION ===================================

#=== MIN_MAX SCALAR ===

X =data_frame.drop(["label"],axis=1)
Y = data_frame['label']

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
df = pd.DataFrame(x_scaled)

print("====================================================")
print("------ Before applying min-max normalization -------")
print("====================================================")
print()
print(X.head(10))

print("====================================================")
print("------ After applying min-max normalization -------")
print("====================================================")
print()
print(df.head(10))


#===================== DATA SPLITTING ====================================

#==== TEST AND TRAIN ====

X_train, X_test, Y_train, Y_test = train_test_split(x_scaled,Y,test_size=0.3,random_state=40)
print("====================================================")
print("------------------ Data Splitting ------------------")
print("====================================================")
print()
print("Total number of data's in input:",data_frame.shape)
print()
print("Total number of data's in training part:",X_train.shape)
print()
print("Total number of data's in testing part:",X_test.shape)
print()

#===================== CLASSIFICATION ====================================

from sklearn import linear_model
from sklearn import metrics

#=== LOGISTIC REGRESSION ===

#initialize the model
logreg = linear_model.LogisticRegression(solver='lbfgs' , C=10)

#fitting the model
logistic = logreg.fit(X_train, Y_train)

#predict the model
y_pred_lr = logistic.predict(X_test)



#=== SVM ===

from sklearn.svm import SVC

svm = SVC(kernel = 'linear')

#fitting the model
svm = svm.fit(X_train[0:5000], Y_train[0:5000])

#predict the model
y_pred_svm = svm.predict(X_test[0:5000])



#=== HYBRID ===
from sklearn.ensemble import VotingClassifier


estimators = []

#=== SVM ====
estimators.append(('svm1', svm))

#=== LOGISTIC REGRESSION ===
estimators.append(('logistic1', logreg))


ensemble = VotingClassifier(estimators)

ensemble.fit(X_train, Y_train)

y_pred_1 = ensemble.predict(X_test)



#======================= PREDICTION =================================

#==== ATTACK DETECTION ====

print("=================================================")
print("------------------  Prediction ------------------")
print("=================================================")

print()
for i in range(1,10):
    if y_pred_1[i]== 11:
        # print("============================")
        print()
        print([i],' Intrusion Attack ')
        print()
        print("============================")
    else:
        # print("============================")
        print()
        print([i],'Non attack ')
        print()
        print("============================")


#======================= PERFORMANCE ANALYSIS =================================

#=== LR  ===

Accu_lr=metrics.accuracy_score(Y_test, y_pred_lr)*100

print("=================================================")
print("------------ 1.Logistic Regression --------------")
print("=================================================")
print()
print("1.Accuracy: ",Accu_lr ,'%')
print()
print(metrics.classification_report(Y_test, y_pred_lr))
print()

#=== SVM ===

Accu_svm=metrics.accuracy_score(Y_test[0:5000], y_pred_svm)*100
print("===================================================")
print("------------ 2. Support Vector Machine ------------")
print("===================================================")
print()
print("1.Accuracy: ",Accu_svm ,'%')
print()
print(metrics.classification_report(Y_test[0:5000], y_pred_svm))
print()

#=== HYBERID LR AND SVM  ===

Accu_hybrid=metrics.accuracy_score(Y_test, y_pred_1)*100
print("===================================================")
print("--------------- 3. Hybrid LR and SVM --------------")
print("===================================================")
print()
print("1.Accuracy: ",Accu_hybrid ,'%')
print()
print(metrics.classification_report(Y_test,y_pred_1))
print()


#======================= COMPARISON  =================================

if (Accu_hybrid>Accu_lr and Accu_hybrid>Accu_svm):
    print("============================")
    print()
    print("    Hybrid is efficient     ")
    print()
    print("============================")

elif Accu_lr>Accu_hybrid:
    print("=================================")
    print()
    print(" Logistic regression is efficient")
    print()
    print("=================================")
else:
    print("============================")
    print()
    print("      SVM is efficient"      )
    print()
    print("============================")








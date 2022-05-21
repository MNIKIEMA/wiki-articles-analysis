from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import set_config
import config as cfg

#VARIABLES
params_logreg = { "vect__ngram_range" : [(1, 1), (1,2)],
                  "tfidf__use_idf" : [True, False], #If true tfidf weigthing, False no weigthing
                  "clf__C" : cfg.clf__C, #penalty parameter 
          }

params_mnb = { 
             'vect__ngram_range' : [(1, 1), (1,2)],
            "tfidf__use_idf" : [True, False], 
            "clf__alpha" : cfg.clf__alpha}

#METHODS
def clf_metric(model, y_test, ypred):
    '''return metrics (accuracy, precision, recall, f1) in a dictionary, and display the confusion matrix''' 
    accuracy = metrics.accuracy_score(y_test, ypred)
    # average is used to avoid 0 division
    precision = metrics.precision_score(y_test, ypred,average='micro')
    recall = metrics.recall_score(y_test, ypred, average='micro')
    f1_scor = metrics.f1_score(y_test, ypred, average='micro')
    scores = {"Accuracy" : accuracy,"Precision" : precision,
            "Recall" : recall, "F1_score" : f1_scor
            }
    confusion_mat = metrics.confusion_matrix(y_test, ypred,
                                           labels = model.classes_)
    #Display the confusion matrix
    mat = metrics.ConfusionMatrixDisplay(confusion_mat,
                        display_labels=model.classes_)
    #plt.figure(figsize=(8,8))
    mat.plot()
    plt.xticks(rotation = 90)
    plt.show()  
    return scores

def accuracy_per_cat( model, y_test, ypred):
  # compute the confusion matrix, and take its diagonal
  #accuracy of each label is the diagonal of the normalized confusion matrix
  mat = metrics.confusion_matrix(y_test, ypred, normalize='true')
  mat = mat.diagonal()
  plt.bar(model.classes_, mat)
  plt.xticks(rotation = 90)
  plt.show()
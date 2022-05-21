'''Modify this file to change the parameters of the code'''

#GLOBAL:
sep = '\t' #separator for your csv files
scrapping = False #True if you want to run the scrapping task, False if you want to skip it
preprocessing = True #True if you want to run the preprocessing task, False if you want to skip it
clustering = True #True if you want to run the clustering task, False if you want to skip it
classification = True #True if you want to run the classification task, False if you want to skip it

#SCRAPPING :
#the list of categories of articles you want to scrap
categories = ['Written_communication','Airports', 'Artists', 'Astronauts', 'Astronomical_objects',
              'Building','City','Comics_characters', 'Companies', 'Foods', 'Monuments_and_memorials',
              'Politicians','Sports_teams','Sportspeople', 'Transport',  'Universities_and_colleges']
k = 100 # number of articles to extract. Beware that scrapping can take a lot of time if you want to scrap a lot of articles 
n= 4 # if an article contains less than n sentences, this article is ignored
savepath = 'scrapping.csv'  #the name of the .csv you want to store your scrapped data,

#PREPROCESSING: 

prep_input_data = 'scrapping.csv'  #input for preprocessing task
lemmatize=False# lemmatize the tokens ? 
#the name of the .csv you want to store your preprocessed data (non-lemmatized)
non_lemmatized_path = 'preprocessed_data.csv'
#the name of the .csv you want to store your preprocessed data (lemmatized)
lemmatized_path = 'preprocessed_data_lemmatized.csv' 

#CLUSTERING, CLASSIFICATION:
cl_input_data =  'preprocessed_data.csv' #input for classification and clustering tasks, must be csv
max_features = 450 #for vectorisation functions (e.g: tfidf)
#number of clusters you choose after analyzing the results of the first and second cluster
#for our dataset it is 10
true_k = 10 
kmeans_max_iter = 700 #nb of max iterations for Kmeans function
clf__C = [1, 5, 8] #penalty parameter  for GridSearchCV function - logistic regression step
clf__alpha = [0.08, 0.1, 0.2] #Laplace smoothing


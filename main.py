'''Do not modifiy this file to change the parameters, please modify config.py instead'''

#LIBRAIRIES
from preprocessing import *
from scrapping import *
from clustering import *
from classification import *
import config as cfg
import pandas as pd
import numpy as np


#METHODS
def scrapping(cat_list=cfg.categories, k = cfg.k, n = cfg.n, savefile=cfg.savepath):
    data = {}
    for c in cat_list:
        print(c)
        i = 10
        enough_articles = False
        while not enough_articles:
            '''while there are not enough enough articles, we are keep scrapping them
            Note that we are erasing articles at each new iteration of the loop, to avoid duplicates'''
            i+=1
            title, text, infobox, wikidata,description = SPARQLQuery(c, k, n,i)
            data[c] = {'title':title, 'text':text, 'infobox':infobox, 'wikidata':wikidata, 'description': description}
            if len(data[c]['title']) ==k:
                enough_articles = True

    print('Scrapping done.')
    
    #print the number of articles for each category
    for category in data:
        print(category, len(data[category]['infobox']))
        
    #save the results 
    save_results_csv(cat_list, data, filename=savefile)
    
def preprocessing(filepath= cfg.prep_input_data, lemmatize = cfg.lemmatize):
    df = pd.read_csv(filepath, sep=cfg.sep) #import the results
    
    # preprocessing of  wikipedia text and wikidata description
    preprocessed_text = pd.DataFrame(df['Text'].apply(lambda x: clean_text(str(x), lemmatize))) 
    preprocessed_desc = pd.DataFrame(df['description'].apply(lambda x:
                                                              clean_text(str(x), lemmatize)))
    
     #Concatenate the results
    data = pd.concat([df['Title'],df['Text'], preprocessed_text, df['description'],
                      preprocessed_desc, df['Category']], axis = 1)
    
    #Rename columns
    data.columns = ['Title', 'Wiki_text', 'Clean_wiki_text',
                    'Wikidata_desc', 'Clean_wikidata_desc', 'Category']
    
    #Export to csv
    if lemmatize:
        data.to_csv(cfg.lemmatized_path , sep=cfg.sep, index = False)
    else:
        data.to_csv(cfg.non_lemmatized_path, sep=cfg.sep, index = False)
    print('Preprocessing done.')
    
    
def clustering():
        
    input('The first attempt of clustering will begin after pressing ENTER')
    dataframe = pd.read_csv(cfg.cl_input_data, sep = cfg.sep)# Import data

    vectorizer = TfidfVectorizer(max_features=cfg.max_features,#Conversion to a matrix of TF-IDF features
                                   use_idf=True,
                                   stop_words='english',
                                   tokenizer=nltk.word_tokenize)#remove stopwords and tokenize words
    X = dataframe['Clean_wiki_text']
    X_vec = vectorizer.fit_transform(X)
    labels = dataframe['Category']
    labels_list = labels.tolist()
    nb_labels = len(dataframe['Category'].unique())
    #First clustering attempt
    print('Number of clusters: {}'.format(nb_labels))
    cluster = KMeans(n_clusters=nb_labels).fit(X_vec)
    score_1 = compute_cluster_metrics(cluster, labels, X_vec)
    for score, value in score_1.items():
        print(score, value)
    _= plot_metrics(cluster, labels, X_vec)
    model = cluster
    cluster_visualization(model, nb_labels, X_vec)
    top_terms_by_cluster(model, np.unique(labels_list).shape[0], vectorizer)
    input('The second attempt of clustering will begin after pressing ENTER')
    #Second clustering attempt
    print('Number of clusters: {}'.format(nb_labels))
    svd = TruncatedSVD(n_components = 10) #Use the first 10 dimension of truncated svd(lsa)
    normalizer = Normalizer(copy=False) #To normalize the input
    cluster_pipe = make_pipeline(svd, normalizer) #build the pipeline
    X_lsa = cluster_pipe.fit_transform(X_vec)
    km =KMeans(n_clusters=nb_labels, max_iter = cfg.kmeans_max_iter).fit(X_lsa)
    score_2 = compute_cluster_metrics(km,labels, X_lsa)
    for score, value in score_2.items():
        print(score, value)
    scores_km = plot_metrics(km, labels, X_lsa)
    cluster_visualization(km, n_clusters=nb_labels, X= X_lsa)
    #Third clustering attempt
    input('The third attempt of clustering will begin after pressing ENTER')
    # get the number of clusters
    true_k = cfg.true_k
    print('Number of clusters: {}'.format(true_k))
    km2 = KMeans(n_clusters=true_k, max_iter=cfg.kmeans_max_iter)
    km2.fit(X_lsa)
    score_3 = compute_cluster_metrics(km2,labels, X_lsa)
    for score, value in score_3.items():
      print(score, value)
    cluster_visualization(km2, true_k, X_lsa)
    print("Top terms per cluster:")

    
# get original from svd
# get the cluster center of each cluster 
# argsort() return the index of each dimension in the cluster center and sort them in increasing value order
# [:, ::-1] reverts the argsort() list to place the indices with highest value first (decreasing order)
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

# terms maps a vectorizer index to the corresponding token
    terms = vectorizer.get_feature_names_out()

# for each cluster
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
    # print out the token of the centroid (order by decreasing tf-idf value)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print('\n')
        
def classification():
    dataframe = pd.read_csv(cfg.cl_input_data, sep=cfg.sep)
    #shuffle the data
    sample_data = dataframe.sample(frac=1)
    #get the features
    features = sample_data['Clean_wiki_text'] 
    #get the labels
    targets = sample_data['Category'] 
    #split our dataset in two subsets
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        targets,
                                                    test_size = 0.2,# 80% for the training and 20% for testing,
                                                    stratify = targets)# stratify to force the distribution of the labels in test and train
    # Logistic regression
    # To visualise the pipeline
    input('Press ENTER to start the logistic regression')
    set_config(display="diagram")
    pipe = Pipeline(    # the pipeline is used to group estimators and works like sklearn estimators
    [
        ("vect", CountVectorizer(max_features= cfg.max_features, #get the frequency of words
                                 stop_words='english',
                                 tokenizer=nltk.word_tokenize)),
        ("tfidf", TfidfTransformer()), #weighting with tfidf
     # logistic regression as a classifier with a solver recommended by sklearn for small dataset
        ("clf", LogisticRegression()), 
    ])
    pipe.fit(X_train, y_train) #fit model to dataset
    ypred = pipe.predict(X_test) #make predictions
    print('Accuracy: {}'.format(metrics.accuracy_score(y_test,ypred))) #compute the accuracy of the test data
    print('F-Measure: {}'.format(metrics.f1_score(y_test,ypred,average='micro'))) #compute f-measure
    metrics_ = clf_metric(pipe, y_test, ypred)
    accuracy_per_cat(pipe, y_test, ypred)
    grid = GridSearchCV(pipe, params_logreg)
    grid.fit(X_train, y_train)
    print('Grid best parameters: {} \n Grid best score: {}'.format(grid.best_params_, grid.best_score_))
    ypred_grid = grid.predict(X_test)
    _ = clf_metric(grid, y_test, ypred_grid)
    input('Press ENTER to finish with the classification with a Multinomial Naive Bayes approach')
    
    #Multinomial Naive Bayes approach
    pipe2 = Pipeline(
    [("vect", CountVectorizer(max_features = cfg.max_features,
                                 stop_words='english', 
                                 tokenizer=nltk.word_tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()), #Multinomial Naive Bayes as classifier
    ])
    
    pipe2.fit(X_train, y_train)
    ypred1 = pipe2.predict(X_test)
    print('Accuracy: {}'.format(metrics.accuracy_score(y_test,ypred1)))
    grid_search = GridSearchCV(pipe2, params_mnb)
    grid_search.fit(X_train, y_train)
    print('Grid best parameters: {} \n Grid best score: {}'.format(grid_search.best_params_,
                                                                   grid_search.best_score_
                                                                   ))
    ypred_1= grid_search.predict(X_test)
    print(clf_metric(grid_search, y_test, ypred_1 ))
          
#MAIN PROGRAM
if __name__ == '__main__':
    #SCRAPPING
    if cfg.scrapping:
        scrapping()
    
    #PREPROCESSING
    if cfg.preprocessing:
        input("Please press the Enter key to run the preprocessing part")
        preprocessing()
    
    #CLUSTERING
    if cfg.clustering:
        input("Please press the Enter key to run the clustering part")
        clustering()
    #CLASSIFICATION
    if cfg.classification:
        input("Please press the Enter key to continue with the classification part")
        classification()
    
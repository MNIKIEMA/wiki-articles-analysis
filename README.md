# Wiki-articles-analysis

The code contains a scrapper, a preprocessing tool, a clustering tool and a classifying tool. 
 - The scrapper consists on scrapping articles from wikipedia, using SPARQL queries combined with Python librairies. 
 - The preprocessing task mainly uses regular expressions with the module re to preprocess texts and nltk module to tokenize texts.
 - The clustering task mainly uses sklearn modules to get metrics and computing clusters. 
 - The classification task mainly uses sklearn again, to classify texts with a logistic regression function, then doing a second classification with a Multinomial Naive Bayes approach.


## Installation
Please run the follow lines in your console terminal. 
Make sure you have at least Python 3.8 and conda installed on your computer. 
>python3 -m pip install --upgrade pip
>pip install -r requirements.txt

You will have to manually install the wptools package, using these following lines.
>conda remove zeromq 
>conda install zeromq 
>conda install conda-build 
>conda install pycurl 
>pip3 install wptools

## Getting started

All the files are contained in a .zip folder: do not split the content of the folder.
To run the code, you can launch the main.py with your Python interpreter, or open your console terminal, move to your directory containing the files and write ```python3 main.py``` in your console.
As long you do not modify the config.py file, it will run the four tasks in a row. 

## Modifying settings

You can modify some settings by opening the config.py file with a text editor. For example, you can set the number of max_features for text_vectorization, or choosing to lemmatize words during the preprocessing step. More classic settings are also available, such as the path or the name of the file to import as input for a specific task, the name of the file used to save your data, or the tasks you want to run. 
After modifying the file, save it, and run the main.py file again. 

## Informations about methods

Some information about the code.

## Scrapping
- A list of categories are set by default :  ['Airports', 'Artists', 'Astronauts', 'Astronomical_objects', 'Building','City','Comics_characters', 'Companies',
             'Foods', 'Monuments_and_memorials','Politicians','Sports_teams','Sportspeople', 'Transport', 
              'Universities_and_colleges', 'Written_communication'] 
- The code will try to find k articles containing at least n sentences. The number of sentences are determined using nltk module. 
- Scrapping is the longest task to run, especially if the number of articles to scrap is high. 

### Preprocessing 
- Tags are removed using regex. 
- Punctuation signs are removed with the string module. 
- Tokenization and all others preprocessing tasks use nltk.

### Clustering
- First, you need to have text and the labels in your input file.
- The text will be vectorized.
- After calling the model, you will get a trained model the number of clusters specified.
- compute_cluster_metrics(), plot_metrics() and cluster_visualization() use the trained model to compute scores.
- The last cluster try to clusterize our data with k clusters ; its value can be set by modifying the value of true_k variable in the config.py file .

### Classification
- Your data is divided train/test for validate (pipeline is recommanded).
- To use, instantiate a classifier and fit the model.
- clf_metric(), accuracy_per_cat() take the model and use test data and the predictions to show scores.

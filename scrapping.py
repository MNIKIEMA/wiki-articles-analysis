#IMPORT LIBRAIRIES

from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia
import wptools
import re
from nltk.tokenize import sent_tokenize
#VARIABLES

k = 10 # number of articles to extract 
n = 3 # if an article contains less than n sentences, this article is ignored

Categories = ['Written_communication','Airports', 'Artists', 'Astronauts', 'Astronomical_objects',
              'Building','City','Comics_characters', 'Companies', 'Foods', 'Monuments_and_memorials',
              'Politicians','Sports_teams','Sportspeople', 'Transport',  'Universities_and_colleges'] 

def SPARQLQuery(category:str, k:int, n:int, broader = 0):
  #variables
    current_number_of_articles=0
    titles,texts, infoboxes, wikidatas, descriptions = [],[],[],[],[]
    wikipedia.set_lang('en')
    prefix1="PREFIX dcterms:<http://purl.org/dc/terms/> "
    prefix2 = "PREFIX dbc:<http://dbpedia.org/resource/Category:> "
    select = 'SELECT ?article WHERE {{?article dcterms:subject/skos:broader{{,{}}} dbc:{} . }}'.format(broader,
                                                                                                       category) 
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(prefix1 + prefix2 +select+'LIMIT {}'.format('1000'))
    #print(broader)
    sparql.setReturnFormat(JSON) 
    ret = sparql.queryAndConvert()
    print(ret)
    for r in ret['results']['bindings']:
        try:
            title = re.sub(r'_', r' ',r['article']['value'].split('/')[-1])
            #print(title)
            page = wptools.page(title, silent = True)
            page.get_query()
            content = re.sub(r'\s+', r' ', page.data['extract'])
            if len(sent_tokenize(content)) >= n:
                #print(title)
                page.get_wikidata()
                if page.data['description']:
                    if page.data['wikidata']:
                        page.get_labels()
                        titles.append(title)
                        texts.append(content)
                        wikidatas.append(page.data['wikidata'])
                        descriptions.append(page.data['description'])
                        page.get_parse()
                        infoboxes.append(page.data['infobox'])
                        current_number_of_articles += 1                    
            if current_number_of_articles == k:
                assert len(infoboxes) == len(titles) 
                assert len(infoboxes)== len(texts)
                assert len(infoboxes) == len(wikidatas)
                assert len(infoboxes)==len(descriptions)
                assert len(infoboxes)== k
                return titles,texts, infoboxes, wikidatas,descriptions
        except Exception as e:
            print(e)
            continue
        
    
        
    print('number of articles = {}'.format(current_number_of_articles))
    return titles,texts, infoboxes, wikidatas, descriptions

def save_results_csv(categories:list, data:dict, filename='scrapping.csv'):
    with open(filename, 'w') as f:
        f.write('Category\tTitle\tText\tInfobox\tWikidata\tdescription\t\n') #Header
        for c in categories:
            for title, text, infobox, wikidata, description in zip(data[c]['title'],data[c]['text'],
                                                      data[c]['infobox'], data[c]['wikidata'], data[c]['description']):
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(c,title,text,infobox, wikidata, description))



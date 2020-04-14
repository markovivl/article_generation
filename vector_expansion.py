import gensim
from tqdm import tqdm
import stanfordnlp
import nltk
from multi_rake import Rake
import numpy as np


def get_text(article):
    """
    Get the original text from the labeled article.
    :param: 
        article: element of the dataset, list of sentences,
            which are lists of strings
    :return: 
        text: str
    """
    text = str()
    for sentence in article:
        for comb in sentence:
            word, label = comb.split()
            text += (word +' ')
    return text
            

def get_keywords(article):
    """
    Find the keywords in article and return them in a convenient way.
    :params:
        article, list of sentences, sentences are lists of strings 
    :returns:
        keywords, list of strings -- extracted keywords
    """
    text = get_text(article)
    text = ''.join(c for c in text if c not in '\'\"')
    rake = Rake()
    try:
        fig = rake.apply(text)
    except:
        fig = get_tokens(article)
        return fig
    keywords = []
    for string, _ in fig:
        keywords += string.split()
        
    return keywords


def get_tokens(article):
    """
    Get list of all words in the text.
    :params:
        article, article, list of sentences, sentences are lists of strings--
        element of the dataset
    :returns:
        tokens, list of strings
    """
    tokens = []
    for sentence in article:
        for comb in sentence:
            word, label = comb.split()
            tokens.append(word)
    return tokens


def get_changes(words, model, closest = 1, word_percent = 1):
    """
    Creates a dict object to be later used for synonimization. 
    :params: 
        words, array of strings -- words that are to be changed
        
        model, KeyedVectors object -- word-vector embedding
        
        closest, int -- closeness of a synonym to find, the bigger the less similar.
        
        word_percent, int 0 < x <= 1 -- percentage of words to change
        
    :returns:
        chagedict, dict, words (str) : synonyms (str) -- dictionary to be used for
        synonimization
    
    """
    if word_percent < 1:
        words = list(np.random.choice(np.array(words), size = int(len(words)*word_percent)))
        
    closest_array = list(np.random.choice(np.array([closest, closest + 1, closest + 2, closest + 3]), size = int(len(words))))
        
    changedict = {} #словарь
    for comb, closest in zip(words,closest_array):
        a = comb.split() #очень часто слов в одной строке более одного, поэтому будем заменять каждое
        for word in a:
            try:
                #если модель знает такое слово, заменяем его
                alt = model.most_similar(positive = ['{}'.format(word)])[closest][0] 
            except:
                alt = word #если модель не знает такого слова, мы его просто оставляем
            changedict[word] = alt #запись слова и синонима в словарь
    return changedict



def synonimize(article, changedict, label= None, postype = None):
    """
    Transform the article according to the previously constructed dictionary.
    :params:
        article, list of sentences, sentence is a list of strings 
        
        changedict, dict -- dictionary with synonyms
        
        label, str or NoneType, values = 'neutral', 'propaganda', None if choose all
        
        postype, list of strings or NoneType, 
        values = 'adj', 'adv', 'v', 'n' -- preferred parts of speech to be 
        included in the synonimization, None for all types
        Example: speechpart = ['adj', 'v', 'adv']
        
    :returns:
        synonimized_article, list of sentences, sentence is a list of strings
    """
    
    #additional variable that holds labels that would NOT be changed
    labeltype = set()
    if label is not None:
        if label == 'neutral':
            labeltype.add('B-SPAN')
            labeltype.add('I-SPAN')
        elif label == 'propaganda':
            labeltype.add('0')
        
    #additional variable that holds parts of speech that would NOT be changed
    pos_shortcuts = {
    'NN': 'n',
    'JJ': 'adj',
    'RB': 'adv',
    'VB': 'v'
}
    wordtypes = set()
    if postype is not None:
        wordtypes = {'n', 'adj', 'adv', 'v'}
        for fig in postype:
            wordtypes.discard(fig)
    
    synonimized_article = []
    for sentence in article:
        synonimized_sentence = []
        for comb in sentence:
            word, label = comb.split()
            pos = nltk.pos_tag([word])[0][1]
            if pos in pos_shortcuts:
                pos = pos_shortcuts[pos]
            if word in changedict and label not in labeltype and pos not in wordtypes:
                new_word = changedict[word]
            else:
                new_word = word
            new_comb = new_word + " " + label
            synonimized_sentence.append(new_comb)
        synonimized_article.append(synonimized_sentence)
    return synonimized_article



def transform_article(article, model, strat = 'keywords', label = None, postype =
                      None, closest = 1, word_percent = 1):
    """
    :params:
        article, article, list of sentences, sentences are lists of strings
        
        strat: str, values are "keywords", "all" -- strategy of synonimizing, 
        rather to choose keywords or all words respectively
        
        label, str or NoneType, values = 'neutral', 'propaganda', None if choose all
        
        postype, list of strings or NoneType, 
        values = 'adj', 'adv', 'v', 'n' -- preferred parts of speech to be 
        included in the synonimization, None for all types
        Example: speechpart = ['adj', 'v', 'adv']
        
        closest, int -- closeness of a synonym to find, the bigger the less similar.
        
        word_percent, int 0 < x <= 1 -- percentage of words to change
        
        word_replace, bool -- calculate word_percent with or without randomization,
                            only relevant when word_percent < 1    
    """
    if strat == 'keywords':
        words = get_keywords(article)
    else:
        words = get_tokens(article)
    changedict = get_changes(words=words, model=model,
                             closest=closest, word_percent=word_percent)
    
    transformed_article = synonimize(article=article, changedict=changedict,
                                    label=label, postype=postype)
    
    return transformed_article
    
    
    


def transform_dataset(dataset, model_dict, model_name = 'glove', strat = 'keywords', 
                      label = None, postype =None, closest = 1, 
                      article_percent= 1,
                      word_percent = 1, num_copies=1):
    """
    :params:
        dataset, list of articles, articles are list of sentences, 
        sentences are lists of strings
        
        model_dict, dict with embedding models
        
        model_name, str, values = 'glove', 'word2vec', 'fasttext'
        
        strat: str, values are "keywords", "all" -- strategy of synonimizing, 
        rather to choose keywords or all words respectively
        
        label, str or NoneType, values = 'neutral', 'propaganda', None if choose all
        
        postype, list of strings or NoneType, 
        values = 'adj', 'adv', 'v', 'n' -- preferred parts of speech to be 
        included in the synonimization, None for all types
        Example: speechpart = ['adj', 'v', 'adv']
        
        closest, int -- closeness of a synonym to find, the bigger the less similar.
        
        article_percent, int 0 < x <=1 -- percentage of articles to transform
        
        word_percent, int 0 < x <= 1 -- percentage of words to change
        
        word_replace, bool -- calculate word_percent randomly with or without replacement,
                            only relevant when word_percent < 1    
    """
    #инициализируем эмбединг
    
    if model_name == 'fast':
        model = model_dict[model_name]
    elif model_name == 'word2vec':
        model = model_dict[model_name]
    else:
        model = model_dict['glove']
        

    #выбираем нужные статьи
    if article_percent < 1:
        candidates = list(np.random.choice(np.array(dataset),size = int(len(dataset)*article_percent)))
    else:
        candidates = dataset
    
    #производим замену
    #dataset_expanded = dataset.copy()
    dataset_new = []
    for article in candidates:
        for i in range(num_copies):
            new_article = transform_article(article=article, model=model, strat=strat, label = label,
                                           postype = postype, closest = closest, word_percent=word_percent)
            dataset_new.append(new_article)
    dataset_expanded = dataset + dataset_new
    
    return dataset_expanded
    
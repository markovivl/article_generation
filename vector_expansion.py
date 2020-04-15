import gensim
from tqdm import tqdm
import stanfordnlp
import nltk
from multi_rake import Rake
import numpy as np


class ArticleGenerator():

	def __init__(self, model_dict, model_name = 'glove', strat = 'keywords', 
                      label = None, postype =None, closest = 1, 
                      article_percent= 1,
                      word_percent = 1, num_copies=1):
	"""
    
    :params:
        model_dict, dict -- contains the models which will be used for
        synonimizing
          
        model_name, str, values = 'glove', 'word2vec', 'fasttext' or other
        
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
		self.model_dict = model_dict
		self.model_name = model_name
		self.strat = strat
		self.label = label
		self.postype = postype
		self.closest = closest
		self.word_percent = word_percent
		self.num_copies = num_copies
		self.model = self.model_dict[self.model_name]


	def get_text(self, article):
			text = str()
	    for sentence in article:
	        for comb in sentence:
	            word, label = comb.split()
	            text += (word +' ')
	    return text


    def get_keywords(self, article):
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


    def get_tokens(self, article):
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


    def get_changes(self, words):
	    """
	    Creates a dict object to be later used for synonimization. 
	    :params: 
	        words, array of strings -- words that are to be change
	        
	    :returns:
	        chagedict, dict, words (str) : synonyms (str) -- dictionary to be used for
	        synonimization
	    
	    """
	    changedict = {}
	    if word_percent < 1:
	        words = list(np.random.choice(np.array(words), size = int(len(words)*self.word_percent)))
	        
	    closest_array = list(np.random.choice(np.array([self.closest, self.closest + 1, self.closest + 2, self.closest + 3]), size = int(len(words))))
	        
	    for comb, closest in zip(words,closest_array):
	        a = comb.split() #очень часто слов в одной строке более одного, поэтому будем заменять каждое
	        for word in a:
	            try:
	                #если модель знает такое слово, заменяем его
	                alt = self.model.most_similar(positive = ['{}'.format(word)])[closest][0] 
	            except:
	                alt = word #если модель не знает такого слова, мы его просто оставляем

	            changedict[word] = alt #запись слова и синонима в словарь

	    return changedict


    def synonimize(self, article, changedict):
	    """
	    Transform the article according to the previously constructed dictionary.
	    :params:
	        article, list of sentences, sentence is a list of strings 

	        chagedict, dict, words (str) : synonyms (str) -- dictionary to be used for
	        synonimization
	        
	    :returns:
	        synonimized_article, list of sentences, sentence is a list of strings

	    """
	    
	    #additional variable that holds labels that would NOT be changed
	    labeltype = set()
	    if self.label is not None:
	        if self.label == 'neutral':
	            labeltype.add('B-SPAN')
	            labeltype.add('I-SPAN')
	        elif self.label == 'propaganda':
	            labeltype.add('0')
	        
	    #additional variable that holds parts of speech that would NOT be changed
	    pos_shortcuts = {
	    'NN': 'n',
	    'JJ': 'adj',
	    'RB': 'adv',
	    'VB': 'v'
	}
	    wordtypes = set()
	    if self.postype is not None:
	        wordtypes = {'n', 'adj', 'adv', 'v'}
	        for fig in self.postype:
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



    def transform_article(self, article):
    """
    :params:
        article, article, list of sentences, sentences are lists of strings
            
    """
    if self.strat == 'keywords':
        words = self.get_keywords(article)
    else:
        words = self.get_tokens(article)

    changedict = get_changes(words=words)
    
    transformed_article = self.synonimize(article=article, changedict=changedict)
    
    return transformed_article


    def transform_dataset(self, dataset):
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

    #производим замену

    dataset_new = []
    for article in dataset:
        for i in range(num_copies):
            new_article = self.transform_article(article=article)
            dataset_new.append(new_article)
    dataset_expanded = dataset + dataset_new
    
    return dataset_expanded


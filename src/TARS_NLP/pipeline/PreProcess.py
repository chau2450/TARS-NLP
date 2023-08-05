
'''
'''

import nltk
from nltk.corpus import stopwords, wordnet
import re
from tqdm import tqdm



class PreProcessing:
    def __init__(self, corpus: str, ngram_start: int , ngram_end: int, lang = 'english', verbose = True) -> None:

        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('wordnet')
        # nltk.download('stopwords')
        self.corpus = corpus
        self.stop_words = set(stopwords.words(lang))
        self.sentence_list: list
        self.sentence_list_re:list = []
        # self.sentence_list_re_stem: list
        self.sentence_list_re_lem:list = []
        self.features: dict = {}
        self.features_index: dict = {}
        self.features_index_word: dict = {}
        self.features_list:list = []
        self.term_freq: any
        self.inverse_doc_freq: any
        self.ngram:tuple = (ngram_start,ngram_end)
        self.vocab_size: int
        self.verbose = verbose
    
    
    @staticmethod
    def __wordnet_pos(tag: str):
        """_summary_

        Args:
            tag (str): string tag given by nltk 

        Returns:
            _type_: _description_
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  


    def __lemmatization(self, sentence: list) -> list:
        """Helper function to get the lemmatized string. 

        Args:
            sentence (str): regularized string

        Returns:
            list: 
        """
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lem_sentence = []
        pos_tags = nltk.pos_tag(sentence)
        for i in range(len(sentence)):
            tag = nltk.pos_tag([pos_tags[i][0]])[0][1]
            lemma = lemmatizer.lemmatize(sentence[i], self.__wordnet_pos(tag))
            lem_sentence.append(lemma)
        
        return lem_sentence
    
    def processCorpus(self, preprocessed: bool) -> None:
        """
        Pseudo Code:

            1) Sentence tokenizer
            2) find regular expression
            3) filter for stop words
            4) find lemmatized sentence
        """

        if preprocessed == False:
            self.sentence_list = nltk.tokenize.sent_tokenize(self.corpus)

        if self.verbose:
            print(f"Regularization, Stopwords and Lemmatization for corpus\n # of sentences = {len(self.sentence_list)}")
        
        for i in tqdm(range(len(self.sentence_list))):
            # print(''.join(self.sentence_list[i]))
            re_sen = re.sub('[^a-zA-Z]',' ',' '.join(self.sentence_list[i])).lower()
            re_sentence = nltk.tokenize.word_tokenize(re_sen)
            filtered_sen_list = [word for word in re_sentence if word not in self.stop_words]
            self.sentence_list_re.append(str(' '.join(filtered_sen_list)))
            self.sentence_list_re_lem.append(self.__lemmatization(filtered_sen_list)) 

        if self.verbose:
            print(f"Creating ngrams: {self.ngram}")
        
        if self.ngram[-1] - self.ngram[0] == 0:
            self.__1ngrams()
        else:
            self.__ngrams()  
        
         


    
    def __ngrams(self) -> None:
        """Get the ngram features

        Args:
            n (int): ngram value supplied by user --> 1,2,3 ...
        """

        for n in tqdm(range(self.ngram[0], self.ngram[1] + 1)):
            for sentence in self.sentence_list_re_lem:
                for i in range(len(sentence) + n - 1):
                    if i >= (len(sentence) - (n + 1)):
                        break
                    if tuple(sentence[i:i+n]) in self.features:
                        self.features[tuple(sentence[i:i+n])] += 1
                    else:
                        self.features[tuple(sentence[i:i+n])] = 1
                        self.features_index[len(self.features) - 1] = [sentence[i:i+n]]
                        self.features_list.append(sentence[i:i+n])
        
        self.vocab_size = len(self.features)
    
    def __1ngrams(self) -> None:
        """Get the ngram features

        Args:
            n (int): ngram value supplied by user --> 1,2,3 ...
        """


        for sentence in self.sentence_list_re_lem:
            for i in range(len(sentence)):
                                  
                if tuple(sentence[i]) in self.features:
                    self.features[tuple(sentence[i])] += 1
                else:
                    self.features[tuple(sentence[i])] = 1
                    self.features_index[len(self.features)] = sentence[i]
                    self.features_index_word[sentence[i]] = len(self.features)
                    self.features_list.append(sentence[i])

        self.vocab_size = len(self.features)


    @property
    def get_ngrams(self) -> dict:
        return self.features
    
    @property
    def get_post_processed(self) -> list:
        return self.sentence_list_re_lem
    

    def post_processing_vars(self) -> [int, dict, list]:
        return self.vocab_size, self.features, self.sentence_list_re_lem, self.features_index
            
        





'''
'''

import nltk
from nltk.corpus import stopwords, wordnet
import re


class PreProcessing:
    def __init__(self, corpus: str, lang = 'english') -> None:
        """Simple preprocessing of text passed as an argument

        Args:
            corpus (str): corpus/body of text
        """
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('wordnet')
        # nltk.download('punkt')
        # nltk.download('stopwords')
        self.corpus = corpus
        self.stop_words = set(stopwords.words(lang))
        self.sentence_list: list
        self.sentence_list_re = []
        # self.sentence_list_re_stem: list
        self.sentence_list_re_lem = []

    
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
    
    def processCorpus(self) -> None:
        """
        Pseudo Code:

            1) Sentence tokenizer
            2) find regular expression
            3) filter for stop words
            4) find lemmatized sentence
        """

        self.sentence_list = nltk.tokenize.sent_tokenize(self.corpus)
        for i in range(len(self.sentence_list)):
            re_sen = re.sub('[^a-zA-Z]',' ',self.sentence_list[i]).lower()
            re_sentence = nltk.tokenize.word_tokenize(re_sen)
            filtered_sen_list = [word for word in re_sentence if word not in self.stop_words]
            self.sentence_list_re.append(str(' '.join(filtered_sen_list)))
            self.sentence_list_re_lem.append(self.__lemmatization(filtered_sen_list))    


    @property
    def get_post_processed(self) -> list:
        return self.sentence_list_re_lem
            
        




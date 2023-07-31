from TARS_NLP.pipeline.PreProcess import PreProcessing
import numpy as np
import math


class TF_IDF(PreProcessing):
    def __init__(self, corpus: str, lang='english') -> None:
        super().__init__(corpus, lang)
    
    @staticmethod
    def __is_sublist(sublist, main_list):
        n = len(sublist)
        for i in range(len(main_list) - n + 1):
            if main_list[i:i+n] == sublist:
                return True
        return False

    @staticmethod
    def __count_sublist_occurrences(main_list, sublist):
        count = 0
        sublist_length = len(sublist)
        
        for i in range(len(main_list) - sublist_length + 1):
            if main_list[i:i+sublist_length] == sublist:
                count += 1
        
        return count
    

    def term_frequency(self) -> None:
        self.term_freq = np.full((len(self.features_list),len(self.sentence_list_re_lem)),0,dtype=float)
        for i in range(len(self.features_list)):
            for j in range(len(self.sentence_list_re_lem)):
                if self.__is_sublist(self.features_list[i], self.sentence_list_re_lem[j]):
                    self.term_freq[i][j] = (self.__count_sublist_occurrences(self.sentence_list_re_lem[j],
                                                                            self.features_list[i]))/(len(self.sentence_list_re_lem[j]) - len(self.features_list[i]) + 1)
        

    def inverse_term_freq(self) -> None:
        self.inverse_doc_freq = np.full((len(self.features),1),0,dtype=float)
        j = 0
        for _ , count in self.features.items():
            self.inverse_doc_freq[j][0] = math.log(len(self.sentence_list_re_lem)/count)
            j += 1

            
    @property
    def get_TF_IDF(self) -> np.array:
        self.term_frequency()
        self.inverse_term_freq()
        return self.term_freq * self.inverse_doc_freq

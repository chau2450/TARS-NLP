from TARS_NLP.pipeline.PreProcess import PreProcessing
import numpy as np
import math
from nltk.corpus import brown
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import warnings

class TF_IDF(PreProcessing):
    def __init__(self, corpus: str, ngram_start: int, ngram_end: int, lang='english') -> None:
        super().__init__(corpus, ngram_start, ngram_end, lang)
    
    @staticmethod
    def is_sublist(sublist, main_list):
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
                if self.is_sublist(self.features_list[i], self.sentence_list_re_lem[j]):
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


class CBOW(PreProcessing):        

    def __init__(self, ngram_start: int , ngram_end: int, lang='english', 
                 window: int = 2, corpus: str | list = brown, 
                 embed_size = 300, epochs:int = 10, verbose:bool = True, model_framework: str = 'keras') -> None:
        super().__init__(corpus, ngram_start, ngram_end, lang, verbose)
        warnings.filterwarnings('ignore')
        self.window: int = window
        self.context_len: int = 2*self.window

        if type(corpus) == list:
            self.sentence_list = corpus 
            self.processCorpus(True)

        else:
            self.processCorpus(False)


        self.window_size = window
        self.context_words: list = []
        self.label_word: list = []
        self.model: any
        self.embed: int = embed_size


    def __gen_context_pairs(self) -> list:
        for sent in tqdm(self.sentence_list_re_lem):
            for i in range(len(sent)):
                start = i - self.window
                end = i + self.window
                if start >= 0 and end < len(sent):
                    self.label_word.append(sent[i])
                    self.context_words.append([sent[j]
                                            for j in range(start, end + 1) if j != i])
                
        
    def word_to_index(self, word):
        for index, word_list in self.features_index.items():
            if word_list[0][0] == word:
                return index
        return None
        

    
    def CBOW_model(self) -> None:
        if self.verbose:
            print("Creating context pairs")

        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            self.__gen_context_pairs()


            # define the model
            self.model = Sequential()
            self.model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embed, input_length=self.context_len))
            self.model.add(Lambda(lambda x: tf.reduce_mean(x, axis=1), output_shape=(self.embed,)))
            self.model.add(Dense(self.vocab_size, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam')
            print(self.model.summary())


            context_indices = [[self.features_index_word[word] for word in context] for context in self.context_words]
            label_indices = [self.features_index_word[word] for word in self.label_word]

            X = pad_sequences(context_indices, maxlen=self.window_size)
            y = tf.keras.utils.to_categorical(label_indices, num_classes=len(self.features_index))

            for epoch in range(1, 8):
                loss = 0.
                i = 0
                for x, target in zip(context_indices, y):
                    i += 1
                    
                    loss += self.model.train_on_batch(np.expand_dims(x, axis=0), np.expand_dims(target, axis=0))
                    if i % 100000 == 0:
                        print('Processed {} (context, word) pairs'.format(i))
                
                print('Epoch:', epoch, '\tLoss:', loss)

            # self.model.fit(X, y, epochs=10, batch_size=256, validation_split=0.2)
            self.model.save('cbow_model.h5')




        
            

    



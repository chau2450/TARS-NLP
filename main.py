from TARS_NLP.logging import logger
import pandas as pd
from TARS_NLP.models.BasicModels import CBOW
from nltk.corpus import brown

# logger.info("This is a test for the logging feature")

with open('data/raw_data/txt_files/file_1.txt') as f:
    corpus_text = f.read()


cbow_model = CBOW(1,1,corpus=list(brown.sents()))


# _,_,out1,out = cbow_model.post_processing_vars()

cbow_model.CBOW_model()

# print(out1)
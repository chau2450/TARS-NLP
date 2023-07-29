from TARS_NLP.logging import logger
import pandas as pd
from TARS_NLP.pipeline.PreProcess import PreProcessing

# logger.info("This is a test for the logging feature")


obj = PreProcessing('''
                    As of my last update in September 2021, NLTK provides support for both word and sentence tokenization, but it does not have a built-in function specifically for paragraph tokenization. Paragraph tokenization is the process of splitting a text into paragraphs or blocks of text based on specific delimiters, such as empty lines or double line breaks.

However, you can achieve paragraph tokenization using custom code or by leveraging other Python libraries that offer more advanced tokenization capabilities. One option is to use regular expressions to split the text into paragraphs based on your desired delimiters. Here's an example of how you can perform paragraph tokenization using regular expressions:
                    ''')

obj.processCorpus()

print(obj.get_post_processed)

print(len(obj.get_post_processed))
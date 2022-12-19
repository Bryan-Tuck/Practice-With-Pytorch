import pandas as pd

import spacy
import collections

from tqdm import tqdm
tqdm.pandas()

nlp = spacy.load("en_core_web_lg")

class TagExtractor:
    def __init__(self, df, text_column):
        """
        Initialize a TagExtractor instance.
        
        Parameters:
            df (pandas DataFrame): the dataframe containing the text data.
            text_column (str): the name of the column in the dataframe containing the text data.
        """
        self.df = df
        self.text_column = text_column

    def extract_tags(self):
        """
        Extract part-of-speech tags from the text data and create new columns in the dataframe for each tag.
        """
        # Tag text and extract tags into a list
        self.df["tags"] = self.df[self.text_column].progress_apply(lambda x: [(tag.text, tag.pos_) 
                                        for tag in nlp(x)])
        
        # Function to count the element of a list
        def lst_count(lst):
            dic_counter = collections.Counter()
            for x in lst:
                dic_counter[x] += 1
            dic_counter = collections.OrderedDict( 
                             sorted(dic_counter.items(), 
                             key=lambda x: x[1], reverse=True))
            lst_count = [ {key:value} for key,value in dic_counter.items() ]
            return lst_count

        # Count tags
        self.df["tags"] = self.df["tags"].progress_apply(lambda x: lst_count(x))

        # Function create new column for each tag category
        def pos_features(lst_dics_tuples, tag):
            """
            Count the number of occurrences of a given tag in a list of dictionaries.
            
            Parameters:
                lst_dics_tuples (list of dictionaries): a list of dictionaries, where each dictionary represents a tag 
                    and its count.
                tag (str): the tag to count.
            
            Returns:
                int: the number of occurrences of the tag in the list.
            """
            if len(lst_dics_tuples) > 0:
                tag_type = []
                for dic_tuples in lst_dics_tuples:
                    for tuple in dic_tuples:
                        type, n = tuple[1], dic_tuples[tuple]
                        tag_type = tag_type + [type]*n
                        dic_counter = collections.Counter()
                        for x in tag_type:
                            dic_counter[x] += 1
                return dic_counter[tag]
            else:
                return 0

        # Extract features
        tags_set = []
        for lst in self.df["tags"].tolist():
             for dic in lst:
                    for k in dic.keys():
                        tags_set.append(k[1])
        tags_set = list(set(tags_set))
        for feature in tags_set:
             self.df["tags_"+feature] = self.df["tags"].progress_apply(lambda x: 
                                     pos_features(x, feature))

df = pd.read_csv('/data/tweets.csv')
tag_extractor = TagExtractor(df, "tweet")
tag_extractor.extract_tags()
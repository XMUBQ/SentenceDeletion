import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import csv

def string2floatlist(score):
    score_list=score.split("__")
    score_list=[float(x) for x in score_list]
    return score_list

def get_data(path,discourse_path='data/discourse_profile/'):
    X=[]
    Y=[]
    discourse_distribution=[]
    meta=[]
    for article in tqdm(os.listdir(path)):
        data=pd.read_csv(path + article,sep='\t',header=None,quoting=csv.QUOTE_NONE)
        data.columns=['sentence','label']
        article_sentence = list(data.sentence)
        article_labels = list(data.label)

        discourse_data = pd.read_csv(discourse_path + article, sep='\t', header=None, quoting=csv.QUOTE_NONE)
        discourse_data.columns = ['sentence', 'label', 'distribution_list']
        discourse_data['score'] = discourse_data.distribution_list.apply(string2floatlist)
        article_discourse_distribution=list(discourse_data.score)

        X.append(article_sentence)
        Y.append(article_labels)
        meta.append(article)
        discourse_distribution.append(article_discourse_distribution)

    return [(X[i],Y[i],discourse_distribution[i],meta[i]) for i in range(len(X))]
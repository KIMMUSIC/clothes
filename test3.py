from distutils.log import debug
import os
from datetime import datetime
from unicodedata import name
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask import session
import numpy as np
from numpy.linalg import norm
import PIL
from ast import literal_eval
import pandas as pd
import numpy as np
import csv as csv_
from ast import literal_eval
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


csv = pd.read_csv('musin.csv', names = ['id', 'category','attrs'], encoding = 'cp949')
row = csv.loc[(csv['id'] == 5)]
print(row)

clothes = pd.read_csv('result.csv', encoding='cp949')
clothes_df = clothes[['number','name','link','tob','cate','season','situation','gender','tag','attr','nouns']]
user_df = row[['id', 'category','attrs']]
cnt_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
clothes_vect = cnt_vect.fit_transform(clothes_df['nouns'].values.astype('str'))
user_vect = cnt_vect.transform(user_df['attrs'].values.astype('str'))
clothes_sim = cosine_similarity(user_vect, clothes_vect)
clothes_sim_idx = (-clothes_sim).argsort()[::]
clothes_sim_idx = clothes_sim_idx[0][0:9]

file_list=[]
for id in clothes_sim_idx:
    file_list.append(str(id))

print(file_list)
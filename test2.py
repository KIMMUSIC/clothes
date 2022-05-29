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

csv = pd.read_csv('musin.csv',names = ['id', 'category','attrs'], encoding = 'cp949')
movies = pd.read_csv('result.csv', encoding='CP949')

movies_df = movies[['number', 'name', 'link', 'cate', 'season', 'situation', 'gender', 'tag', 'attr', 'nouns']]
clothes_df = csv[['id', 'category','attrs']]

cnt_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
# fit_transform안에 데이터프레임형태로 넣어주면 안됨. 하나의 변수씩만 넣어주자!

print(movies_df['nouns'])
genres_vect = cnt_vect.fit_transform(movies_df['nouns'].values.astype('str'))
genres_vect2 = cnt_vect.transform(clothes_df['attrs'].values.astype('str'))

#keywords_vect = cnt_vect.fit_transform(movies_df['keywords_literal'])

# 장르에 따른 영화별 코사인 유사도 추출
genre_sim = cosine_similarity(genres_vect2, genres_vect)
# 3개만 유사도행렬값 추출해보기
#print(genre_sim[:6])

# argsort를 이용해서 유사도가 높은 영화들의 index 추출
genre_sim_idx = (-genre_sim).argsort()[::]
print(genre_sim_idx[0][0:9])


    
#similar_movies = find_sim_movie(movies_df, genre_sim_idx, csv.loc[(csv['id'] == 5)]['attrs'])
#print(similar_movies[['number', 'name', ]])

#print(csv.loc[(csv['id'] == 5)]['attrs'].tolist())
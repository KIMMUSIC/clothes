from distutils.log import debug
import os
from datetime import datetime
from unicodedata import name
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash
from ast import literal_eval
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
warnings.filterwarnings(action='ignore')
movies = pd.read_csv('result.csv', encoding='CP949')


movies_df = movies[['number', 'name', 'link', 'cate', 'season', 'situation', 'gender', 'tob', 'nouns']]




UPLOAD_FOLDER = '\static\image'
ALLOWED_EXTENSION = {'txt', 'png', 'jpg', 'jpeg', 'gif'}

path = "static/image/"
file_list = os.listdir(path)
count = len(file_list)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///imgname.db' #가상의 db생성 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class dbimg(db.Model):
    id = db.Column('img_id', db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    
    def __init__(self, name):
        self.name = name


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userid = db.Column(db.String(100))
    password = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    
    
    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class Clothes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(10))
    name = db.Column(db.String(200))
    link = db.Column(db.String(200))
    tob = db.Column(db.String(10))
    cate = db.Column(db.String(30))
    season = db.Column(db.String(10))
    situation = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    nouns = db.Column(db.String(200))
    

if __name__ == '__main__':
    db.create_all()
    for i in range(0,3509):
        Clothestable = Clothes()
        Clothestable.number = str(movies_df['number'][i])
        Clothestable.link = movies_df['link'][i]
        Clothestable.tob = movies_df['tob'][i]
        Clothestable.cate = movies_df['cate'][i]
        Clothestable.season = movies_df['season'][i]
        Clothestable.situation = movies_df['situation'][i]
        Clothestable.gender = movies_df['gender'][i]
        Clothestable.name = movies_df['name'][i]
        Clothestable.nouns = movies_df['nouns'][i]

        db.session.add(Clothestable)
        db.session.commit()



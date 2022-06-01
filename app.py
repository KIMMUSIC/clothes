from distutils.log import debug
import os
from datetime import datetime
from unicodedata import name
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import false
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import PIL
from ast import literal_eval
import pandas as pd
import numpy as np
import warnings
import csv as csv_
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from flask import request, make_response
warnings.filterwarnings(action='ignore')
import datetime
import json
with open('result.json', 'r', encoding='utf8') as f:
    json_data = json.load(f)

now=datetime.datetime.now()

# 파이썬은 &&기호로 연결할 필요 없이 두 개의 범위 구분을 한 번에 쓸 수 있다.
if 3<=now.month<=5:
    nseason="spring"

if 6<=now.month<=8:
    nseason="summer"

if 9<=now.month<=11:
    nseason="spring"

if now.month<=2 or now.month==12:
    nseason="winter"

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

import scipy.sparse as sparse
import implicit
import pandas as pd

def init_CF():
    PATH = 'rating.csv'
    file = pd.read_csv(PATH)

    train_data = file.pivot_table('Rating', index='UID', columns='Number').fillna(0)
    temp = sparse.csr_matrix(train_data)
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=900)


    model.fit(temp)
    return temp, model



cf_table, model = init_CF()


def recommand_cf(model, uid, cf_table):
    recommanded = model.recommend(uid, cf_table[0])
    return recommanded[0]



def find_sim_clothes(df, sorted_idx, item_number, top_n=10):
    title_clothes = df[df['number'] == item_number]
    title_clothes_idx = title_clothes.index.values
    top_sim_idx = sorted_idx[title_clothes_idx, :top_n]
    top_sim_idx = top_sim_idx.reshape(-1,)
    similar_clothes = df.iloc[top_sim_idx]
    
    return similar_clothes

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
    
    # def __init__(self,  userid, password, gender):
    #     self.userid = userid
    #     self.password = password
    #     self.gender = gender
    
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

@app.route('/')
def main():
    userid = session.get('userid',None)
    return render_template('index.html', userid = userid)

@app.route('/mycsv', methods=['POST'])
def my_csv():
    itemid = request.form['itemid']
    id = session.get('userid',None)
    item = Clothes.query.filter(Clothes.number == itemid).first()
    attrs = item.nouns
        
    csv = pd.read_csv('musin.csv', names = ['id', 'attrs'], encoding = 'cp949')
    
    find_row = csv.loc[(csv['id'] == int(id))]
    find_row_list = find_row.values.tolist()
    
    print(find_row)
    
    # 출력 부분 print(find_row_list, file=sys.stderr)
    
    # 만약 검색한 값이 없다면 -> 새로운 행 추가( id, category 로 검색 )
    if len(find_row_list) == 0:
        with open('musin.csv' , 'a' , encoding = 'cp949' ,newline = '') as input_file:
            f = csv_.writer(input_file)
            f.writerow([id,attrs])
            
    # 검색해서 값이 나왔다면 -> 처음부터 행들을 저장하면서 수정해야 하는 행이 나오면 수정해서 리스트에 저장.
    else :
        modified_file = []
        with open('musin.csv', 'r', encoding = 'cp949', newline='')as r_file:
            rdr = csv_.reader(r_file)
            
            for line in rdr:
                if (line[0] == id):
                    temp_attrs = find_row_list[0][1]    
                    new_attrs = temp_attrs + ' ' + attrs
                    line[1] = new_attrs
                modified_file.append(line)
                
        with open('musin.csv', 'w', encoding = 'cp949', newline='')as w_file:
            wr =csv_.writer(w_file)
            wr.writerows(modified_file)
               
    return redirect('/')

@app.route('/button_tem')
def button_tem():
    return render_template('button_tem.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/search')
def search():
    return render_template('search.html')

   
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/select_image')
def select_image():
    file_list = os.listdir(path)
    count = len(file_list)
    return render_template('select_image.html', file_list=file_list, count=count, dbimg = dbimg.query.all())


@app.route('/recommend')
def recommend():

    userid = session.get('userid',None)
    user = User.query.filter(User.userid == str(userid)).first()
    ugender = user.gender
    csv = pd.read_csv('musin.csv', names = ['id','attrs'], encoding = 'cp949')
    row = csv.loc[(csv['id'] == int(userid))]
    season = request.cookies.get('season')
    gender = request.cookies.get('gender')
    situation = request.cookies.get('situation')

    clothes = pd.read_csv('result.csv', encoding='cp949')
    clothes_df = clothes[['number','name','link','tob','cate','season','situation','gender','tag','attr','nouns']]
    user_df = row[['id','attrs']]
    cnt_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
    clothes_vect = cnt_vect.fit_transform(clothes_df['nouns'].values.astype('str'))
    user_vect = cnt_vect.transform(user_df['attrs'].values.astype('str'))
    clothes_sim = cosine_similarity(user_vect, clothes_vect)
    clothes_sim_idx = (-clothes_sim).argsort()[::]
    clothes_sim_idx = clothes_sim_idx[0]
    total = 0
    file_list=[]
    c = {}
    c2 = {}
    total2 = 0
    for id in clothes_sim_idx:
        item = Clothes.query.filter(Clothes.number == int(id)).first()
        if season == 'true':
            if item.season != nseason:
                continue
        if gender == 'true':
            if item.gender != '공용':
                if item.gender != ugender:
                    continue

        cate = item.cate
        if cate in c:
            c[cate]  = c[cate] + 1
        else:
            c[cate] = 0
        if(c[cate] > 3): continue
        file_list.append(str(id))
        total = total+1
        if(total >= 10): break



    recommand_item = recommand_cf(model, int(userid), cf_table)
    print(recommand_item)
    for id in recommand_item:
        item = Clothes.query.filter(Clothes.number == int(id)).first()
        print(season, nseason, item.season)
        if season == 'true':
            if item.season != nseason:
                continue
        if gender == 'true':
            if item.gender != '공용':
                if item.gender != ugender:
                    continue
        cate = item.cate
        if cate in c2:
            c2[cate]  = c2[cate] + 1
        else:
            c2[cate] = 0
        if(c2[cate] > 3): continue
        file_list.append(str(id))
        total2 = total2+1
        if(total2 >= 10): break


    return render_template('mainview2.html', file_list=file_list, count=len(file_list))

@app.route('/recommend2')
def recommend2():
    userid = session.get('userid',None)
    user = User.query.filter(User.userid == str(userid)).first()
    ugender = user.gender
    gender = request.args.get('gender')
    season = request.args.get('season')
    situation = request.args.get('situation')
    userid = session.get('userid',None)
    csv = pd.read_csv('musin.csv', names = ['id','attrs'], encoding = 'cp949')
    row = csv.loc[(csv['id'] == int(userid))]

    clothes = pd.read_csv('result.csv', encoding='cp949')
    clothes_df = clothes[['number','name','link','tob','cate','season','situation','gender','tag','attr','nouns']]
    user_df = row[['id','attrs']]
    cnt_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
    clothes_vect = cnt_vect.fit_transform(clothes_df['nouns'].values.astype('str'))
    user_vect = cnt_vect.transform(user_df['attrs'].values.astype('str'))
    clothes_sim = cosine_similarity(user_vect, clothes_vect)
    clothes_sim_idx = (-clothes_sim).argsort()[::]
    clothes_sim_idx = clothes_sim_idx[0]
    total = 0
    file_list=[]
    c = {}
    c2 ={}
    total2 = 0
    for id in clothes_sim_idx:
        item = Clothes.query.filter(Clothes.number == int(id)).first()
        print(season, nseason, item.season)
        if season == 'true':
            if item.season != nseason:
                continue
        if gender == 'true':
            if item.gender != '공용':
                if item.gender != ugender:
                    continue
        cate = item.cate
        if cate in c:
            c[cate]  = c[cate] + 1
        else:
            c[cate] = 0
        if(c[cate] > 3): continue
        file_list.append(str(id))
        total = total+1
        if(total >= 10): break



    recommand_item = recommand_cf(model, int(userid), cf_table)
    print(recommand_item)
    for id in recommand_item:
        item = Clothes.query.filter(Clothes.number == int(id)).first()
        print(season, nseason, item.season)
        if season == 'true':
            if item.season != nseason:
                continue
        if gender == 'true':
            if item.gender != '공용':
                if item.gender != ugender:
                    continue
        cate = item.cate
        if cate in c2:
            c2[cate]  = c2[cate] + 1
        else:
            c2[cate] = 0
        if(c2[cate] > 3): continue
        file_list.append(str(id))
        total2 = total2+1
        if(total2 >= 10): break

    res = make_response(render_template('mainview2.html', file_list=file_list, count=len(file_list)))


    if gender is None: res.set_cookie('gender','false') 
    else: res.set_cookie('gender',gender)
    if season is None: res.set_cookie('season','false')
    else: res.set_cookie('season',season)
    if situation is None: res.set_cookie('situation','false')
    else: res.set_cookie('situation',situation)

    return res
    #return render_template('mainview2.html', file_list=file_list, count=len(file_list))

@app.route('/register', methods=['POST']) #GET(정보보기), POST(정보수정) 메서드 허용
def register2():
    userid = request.form.get('userid')
    gender = request.form.get('gender')
    password = request.form.get('password')
    print(userid, gender, password)
    if not(userid and gender and password):
        return "입력되지 않은 정보가 있습니다"
    else:
        usertable = User()
        usertable.userid = userid
        usertable.gender = gender
        usertable.password = password
        print(userid, gender, password)
        db.session.add(usertable)
        db.session.commit()
        return "회원가입 성공"
    return redirect('/')

    

@app.route('/login', methods=['POST']) #GET(정보보기), POST(정보수정) 메서드 허용
def login2():
    userid = request.form.get('userid')
    password = request.form.get('password')
    
    session['userid'] = userid
    return redirect('/')

@app.route('/query', methods=['POST']) #GET(정보보기), POST(정보수정) 메서드 허용
def query():
    query = request.form.get('query')
    search = "%{}%".format(query)
    id = Clothes.query.filter(Clothes.nouns.like(search)).all()
    file_list=[]
    for num in id:
        file_list.append(str(num.number)+".png")
    return render_template('query.html', file_list=file_list, count=len(file_list), dbimg = dbimg.query.all())
    

    
@app.route('/single_move')
def single_move():
    return render_template('single_move.html')

@app.route('/upload')
def render_file():
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSION

@app.route('/index', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
        # 저장할 경로 + 파일명
            f.save("./static/image/"+secure_filename(f.filename))
            temp = secure_filename(f.filename)
            imagename = dbimg(f.filename)
            db.session.add(imagename)
            db.session.commit()
            return render_template('index.html', Testname = temp)
    return render_template('index.html' , error = "사진 파일 에러")

@app.route('/mainview/<category>', methods = ['GET'])
def goods(category):
    query = category
    id = Clothes.query.filter(Clothes.cate == query).all()
    file_list=[]
    for num in id:
        file_list.append(str(num.number)+".png")
    return render_template('mainview.html', file_list=id, count=len(file_list))

@app.route('/detail/<number>', methods = ['GET'])
def detail(number):
    query = number
    id = Clothes.query.filter(Clothes.number == query).first()
    attrs = json_data[int(number)]['tag']

    clothes = pd.read_csv('result.csv', encoding='cp949')
    clothes_df = clothes[['number','name','link','tob','cate','season','situation','gender','tag','attr','nouns']]
    cnt_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
    clothes_vect = cnt_vect.fit_transform(clothes_df['nouns'].values.astype('str'))
    clothes_sim = cosine_similarity(clothes_vect, clothes_vect)
    clothes_sim_idx = (-clothes_sim).argsort()[::]
    clothes_sim_idx = clothes_sim_idx[int(query)][1:7]
    file_list=[]
    for a in clothes_sim_idx:
        cl = Clothes.query.filter(Clothes.number == int(a)).first()
        file_list.append(cl)



    return render_template('detail.html', id=id,attr = attrs, clothes = file_list, count=len(file_list))


@app.route('/query2', methods = ['POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            f.save(secure_filename(f.filename))
            feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
            filenames = pickle.load(open('filenames.pkl','rb'))
            model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
            model.trainable = False
            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])
            img = image.load_img(f.filename,target_size=(224,224))
            img_array = image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            result = model.predict(preprocessed_img).flatten()
            normalized_result = result / norm(result)

            neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
            neighbors.fit(feature_list)

            distances,indices = neighbors.kneighbors([normalized_result])
            file_list=[]
            for file in indices[0][1:6]:
                print(filenames[file], filenames[file][8:])
                file_list.append((filenames[file])[8:])
            return render_template('query.html', file_list=file_list, count=len(file_list), dbimg = dbimg.query.all())
                

@app.route('/new', methods = ['GET', 'POST'])
def new():
    if request.method == 'POST':
        imagename = dbimg(request.form['name'])
         
        db.session.add(imagename)
        db.session.commit()
        flash('Record was successfully added')
        return redirect(url_for('select_image'))
    return render_template('new.html')
 
@app.route('/remove', methods = ['GET', 'POST'])
def remove():
    if request.method == 'POST':
        imagename = dbimg.query.filter_by(name = request.form['name']).first()
        db.session.delete(imagename)
        db.session.commit()
        return redirect(url_for('select_image'))
    return render_template('remove.html')
 
if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=8080, debug = True)

from distutils.log import debug
import os
from datetime import datetime
from unicodedata import name
from flask import Flask, render_template, request, redirect, url_for, flash
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

@app.route('/select_image')
def select_image():
    file_list = os.listdir(path)
    count = len(file_list)
    return render_template('select_image.html', file_list=file_list, count=count, dbimg = dbimg.query.all())

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

@app.route('/goods/<category>', methods = ['GET'])
def goods(category):
    query = category
    id = Clothes.query.filter(Clothes.cate.like(query)).all()
    file_list=[]
    for num in id:
        file_list.append(str(num.number)+".png")
    return render_template('query.html', file_list=file_list, count=len(file_list), dbimg = dbimg.query.all())


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

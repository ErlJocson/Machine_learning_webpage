from .makePredict import heartAttackPrediction, lungCancerPrediction, titanicDataset
from flask import render_template, url_for, request, redirect
from flask_login import login_user, current_user, logout_user, login_required
from .forms import RegistrationForm, LoginForm
from flask.helpers import flash
from . import app, db, bcrypt
from .models import User
import numpy as np

@app.route('/')
@app.route('/home')
@login_required
def home():
    return render_template("/index.html", title="Home")

@app.route('/about')
@login_required
def about():
    return render_template("about.html", title="About")

@app.route('/products')
@login_required
def products():
    return render_template("products.html", title="Products")

@app.route('/register', methods = ["POST","GET"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email = form.email.data, password = hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created', 'success')
        return redirect(url_for('home'))
    return render_template("register.html", title="Register", form = form)

@app.route('/login', methods = ["POST","GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('home'))
        else:
            flash('Login failed', 'danger')
    return render_template("login.html", title="Login", form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/products/heart-attack', methods=["POST", "GET"])
@login_required
def heartAttact():
    if request.method == "POST":
        sex = request.form["gender"]
        thal_rate = request.form["thal_rate"]
        caa = request.form["caa"]
        oldpeak = request.form["oldpeak"]
        exng = request.form["exng"]
        chest_pain = request.form["chest_pain"]
        fbs = request.form["fbs"]
        restecg = request.form["restecg"]
        slp = request.form["slp"]
        data = [sex,exng,caa,chest_pain,fbs,restecg,slp,thal_rate,oldpeak]
        prediction = heartAttackPrediction(np.array([data]))
        return render_template('Machine-learning-algorithm/heart-attack.html', title="Heart attack analysis", post=prediction)
    else:
        return render_template('Machine-learning-algorithm/heart-attack.html', title="Heart attack analysis")

@app.route('/products/lung-cancer', methods = ["POST","GET"])
@login_required
def lungCancer():
    if request.method == "POST":
        age = request.form["age"]
        smoke = request.form["smoke"]
        data = [age, smoke]
        prediction = lungCancerPrediction(np.array([data]))
        return render_template('Machine-learning-algorithm/lung-cancer.html', title="Lung cancer analysis", post=prediction)
    else:
        return render_template('Machine-learning-algorithm/lung-cancer.html', title="Lung cancer analysis")

@app.route('/products/brain-tumor')
@login_required
def brainTumor():
    return render_template('Machine-learning-algorithm/brain-tumor.html', title="Brain tumor analysis")

@app.route('/products/movie-recommendation', methods=["POST", "GET"])
@login_required
def movieRecommendation():
    return render_template('Machine-learning-algorithm/movie-recommendation.html', title='Movie recommendation')

@app.route('/products/music-recommendation', methods=["POST", "GET"])
@login_required
def musicRecommendation():
    return render_template('Machine-learning-algorithm/music-recommendation.html', title='Music recommendation')

@app.route('/products/anime-recommendation', methods=["POST", "GET"])
@login_required
def animeRecommendation():
    return render_template('Machine-learning-algorithm/anime-recommendation.html', title='Anime recommendation')

@app.route('/products/house-price-prediction', methods=["POST", "GET"])
@login_required
def housePrediction():
    return render_template('Machine-learning-algorithm/house-price.html', title="House price prediction")

@app.route('/products/lot-price-prediction', methods=["POST", "GET"])
@login_required
def lotPrediction():
    return render_template('Machine-learning-algorithm/lot-price.html', title="Lot price prediction")

@app.route('/products/titanic', methods=["POST", "GET"])
@login_required
def titanicPrediction():
    if request.method == "POST":
        Pclass = int(request.form["Pclass"])
        sex = request.form["gender"]
        embark = request.form["embark"]
        data = [Pclass, sex, embark]
        prediction = titanicDataset(np.array([data]))
        return render_template('Machine-learning-algorithm/titanic.html', title="Titanic", post=prediction)
    else:
        return render_template('Machine-learning-algorithm/titanic.html', title="Titanic")
from flask import Flask, render_template, request,redirect,session
from datetime import datetime
app = Flask(__name__) 

@app.route("/", methods = ['get'])
def get_home():
    return render_template('index.html')

# flaskアプリを動かすための記述
if __name__ == "__main__":
    app.run(debug = True)
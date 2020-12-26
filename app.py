from flask import Flask, render_template, request, redirect, session
from datetime import datetime
import os


app = Flask(__name__)

@app.route('/', methods=["GET"])
def get_home():
    return render_template('index.html')

@app.route('/', methods=["post"])
def get_face():
    # ファイル容量を1MBに制限
    app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

    upload = request.files['upload']
    if not upload.filename.lower().endswitch(('.png', 'jpg', 'jpeg')):
        return 'png, jpg, jpeg形式のファイルを選択してください。'

    save_path = get_save_path()
    print(save_path)
    filename = upload.filename
    print(filename)
    upload.save(os.path.join(save_path, filename))

    return redirect('/')

def get_save_path():
    path_dir = "./static/img"
    return path_dir


# flaskアプリを動かすための記述
if __name__ == "__main__":
    app.run(debug = True)
import os
from threading import Thread
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, send
from ai import embed_document, query
from langchain.memory import ConversationBufferMemory

def embed_doc(file):
    embedding_thread = Thread(target=embed_document, args=(file,))
    embedding_thread.start()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

UPLOAD_FOLDER = 'docs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        embed_doc(file=file_path)
        
        return f"File uploaded successfully: {file.filename}"
    return redirect(url_for('index'))

@socketio.on('message')
def handle_message(msg):
    print(f"Message: {msg}")

    response = query(msg)

    send(response, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
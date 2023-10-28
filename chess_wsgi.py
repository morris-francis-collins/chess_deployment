from flask import Flask, request, render_template, jsonify
from chess_predict import chess_predict

import torch
import chess
import chess.pgn
import io
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chess')
def chess():
    return render_template('home.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    global model
    pgn = request.form['pgn']
    word  = request.args.get('pgn')
    #pgn = "1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 5.Ng3 Bg6 6.h4 h6 7.Nf3 Nd7 8.h5 Bh7 9.Bd3 Bxd3 10.Qxd3 Ngf6 11.Bf4 e6 12.O-O-O Be7 13.Ne4 O-O 14.Kb1"
    #pgn = "1.e4 c6 2.d4 d5"
    result = chess_predict(pgn)
    #result = {"X":{"move":"a", "rating":"b"},"Y":{"move":"c","rating":"d"}}
    #result = [{"move":"a","rating":"b"}]
    #result = {"Index" : ["1","2"]}
    #result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

'''
if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True)
'''

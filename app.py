from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime as dt
import datetime
import plotly.express as px

app = Flask(__name__)

# method to predict cryptocurrency price and render the plot on home/predict
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        crypto = request.form.get('crypto')
        start = request.form.get('start')  #%y-%m-%d
        end = request.form.get('end')     #%y-%m-%d
        #print(crypto,end,end)

        if crypto == "bitcoin":
            df_btc = pd.read_csv('dataset/BTC-USD.csv')
            diff = df_btc['High'].max() - df_btc['High'].min()
            reconstructed_model = tf.keras.models.load_model("models/BTC.h5")

        elif crypto == "ethereum":
            df_btc = pd.read_csv('dataset/ETH-USD.csv')
            diff = df_btc['High'].max() - df_btc['High'].min()
            reconstructed_model = tf.keras.models.load_model("models/ETH.h5")

        elif crypto == "usdt":
            df_btc = pd.read_csv('dataset/USDT-USD.csv')
            diff = df_btc['High'].max() - df_btc['High'].min()
            reconstructed_model = tf.keras.models.load_model("models/USDT.h5")

        diff = diff*0.0001
        start_date, end_date = start, end
        n_steps = 7
        
        if start_date not in df_btc['Date'].to_list():
          res = (dt.strptime(end_date, "%Y-%m-%d") - dt.strptime(df_btc['Date'].to_list()[-1], "%Y-%m-%d")).days + 1
          close_last_7 = df_btc[-7:]['Close'].to_list()
          print(df_btc[-7:]['Date'].to_list())
          prediction = []
          dates = []
          previous_date = dt.strptime(df_btc['Date'].to_list()[-1], "%Y-%m-%d") - datetime.timedelta(days=1)
        
          while res != 0:
            close_last_7_Array = np.array(close_last_7).reshape(1, n_steps, 1)
            val = reconstructed_model.predict(close_last_7_Array, verbose=0)
            close_last_7.remove(close_last_7[0])
            close_last_7.append(val[0][0])
            prediction.append(val[0][0])
            previous_date += datetime.timedelta(days=1)
            dates.append(previous_date)
            res -= 1
        
          dates = [str(i).split(' ')[0].strip() for i in dates]
          print(dates.index(start_date))
          index = dates.index(start_date)
          dates = dates[index:]
          prediction = prediction[index:]
        
          fig = go.Figure()
        
          fig.add_trace(go.Scatter(x=dates, y=prediction, mode='lines+markers', name='predicted price', line_color='rgb(0,176,246)'))
          fig.write_html("templates/file.html")        
          
          
        else:
          res = (dt.strptime(end_date, "%Y-%m-%d") - dt.strptime(start_date, "%Y-%m-%d")).days + 1
          print(res, "$$$$$$$$$$$")
          idx = df_btc['Date'].to_list().index(start_date)
          close_last_7 = df_btc['Close'].to_list()[idx-n_steps:idx]
          prediction = []
          dates = []
          previous_date = dt.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(days=1)
        
          while res != 0:
            close_last_7_Array = np.array(close_last_7).reshape(1, n_steps, 1)
            val = reconstructed_model.predict(close_last_7_Array, verbose=0)
            close_last_7.remove(close_last_7[0])
            close_last_7.append(val[0][0])
            prediction.append(val[0][0])
            previous_date += datetime.timedelta(days=1)
            dates.append(previous_date)
            res -= 1
        
          dates = [str(i).split(' ')[0].strip() for i in dates]
          index = dates.index(start_date)
          dates = dates[index:]
          prediction = prediction[index:]        
        
        
          date_acutal = df_btc['Date'].to_list()
        
          if end_date <= date_acutal[-1]:
            end_idx = date_acutal.index(end_date)
          else:
            end_idx = date_acutal.index(date_acutal[-1])
        
          date_acutal = df_btc['Date'][idx:end_idx+1].to_list()
          price_acutal = df_btc['Close'][idx:end_idx+1].to_list()
        
          fig = go.Figure()
        
          fig.add_trace(go.Scatter(x=dates, y=prediction, mode='lines+markers', name='predicted price', line_color='rgb(0,176,246)'))
          fig.add_trace(go.Scatter(x=date_acutal, y=price_acutal, mode='lines+markers', name='actual price', line_color='rgb(231,107,243)'))
          fig.write_html("templates/file.html")        
        
    return render_template("file.html")
        


# Home page that is rendered for every web call
@app.route("/")
def home():
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)

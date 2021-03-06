import asyncio
from aiohttp import web
import socketio
import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

estimator = None
testing_set = None

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)


async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

def parseString(content):
    mystring = content.replace('\n', ' ').replace('\r', '')
    mystring = re.sub(re.compile('<{.*?}>'),"", mystring)
    mystring = re.sub(r'\d+', '', mystring)
    mystring = re.sub('[^ a-zA-Z0-9]', ' ', mystring)
    return mystring.lower()

@sio.on('classify', namespace='/test')
async def test_message(sid, message):
    content = parseString(message['data'])
    testing_ele = pd.DataFrame({'content':[content]})
    predict_ele_input_fn = tf.estimator.inputs.pandas_input_fn(testing_ele, shuffle=False)
    prediction = estimator.predict(predict_ele_input_fn)
    class_predict = list(prediction)[0]['class_ids'][0]
    if class_predict == 0:
        result = 'nspam'
    else:
        result = 'spam'
    await sio.emit('my response', {'data': result}, room=sid,
                   namespace='/test')



@sio.on('test', namespace='/test')
async def test_message(sid, message):
    content = parseString(message['data'])
    testing_ele = pd.DataFrame({'content':[content]})
    predict_ele_input_fn = tf.estimator.inputs.pandas_input_fn(testing_ele, shuffle=False)
    prediction = estimator.predict(predict_ele_input_fn)
    class_predict = list(prediction)[0]['class_ids'][0]
    if class_predict == 0:
        result = 'Your message was not classified as spam.';
    else:
        result = 'Your message was classified as spam!';
    await sio.emit('test response', {'data': result}, room=sid,
                   namespace='/test')

@sio.on('connect', namespace='/test')
async def test_connect(sid, environ):
    await sio.emit('my response', {'data': 'Connected', 'count': 0}, room=sid,
                   namespace='/test')


@sio.on('disconnect', namespace='/test')
def test_disconnect(sid):
    print('Client disconnected')



def initClassifier():
    print("init")
    global estimator
    global testing_set
    data = pd.read_csv("final_data_parsed.csv")
    training_set = data.head(int(len(data) * 0.8))
    testing_set = data.tail(int(len(data) * 0.2))
    embedded_text_feature_column = hub.text_embedding_column( key="content", module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100], feature_columns=[embedded_text_feature_column],n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    model_dir='models/spam')

app.router.add_static('/static', 'static')
app.router.add_get('/', index)


if __name__ == '__main__':
    initClassifier()
    web.run_app(app)

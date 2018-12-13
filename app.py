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


async def background_task():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        await sio.sleep(10)
        count += 1
        await sio.emit('my response', {'data': 'Server generated event'},
                       namespace='/test')


async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('my event', namespace='/test')
async def test_message(sid, message):
    #try:
    num = int(message['data'])
    print(testing_set)
    testing_ele = pd.DataFrame('content',[testing_set['content'][num]])
    predict_ele_input_fn = tf.estimator.inputs.pandas_input_fn(testing_ele)
    prediction = estimator.predict(predict_test_ele_fn)
    print(prediction)
#    except KeyError:
#        print("key error")
#        print(message, message.get('data', None), type(message.get('data', None)))
#    except ValueError:
#        print(message['data'])
#        print("not an int")

    await sio.emit('my response', {'data': message['data']}, room=sid,
                   namespace='/test')


@sio.on('my broadcast event', namespace='/test')
async def test_broadcast_message(sid, message):
    await sio.emit('my response', {'data': message['data']}, namespace='/test')


@sio.on('join', namespace='/test')
async def join(sid, message):
    sio.enter_room(sid, message['room'], namespace='/test')
    await sio.emit('my response', {'data': 'Entered room: ' + message['room']},
                   room=sid, namespace='/test')


@sio.on('leave', namespace='/test')
async def leave(sid, message):
    sio.leave_room(sid, message['room'], namespace='/test')
    await sio.emit('my response', {'data': 'Left room: ' + message['room']},
                   room=sid, namespace='/test')


@sio.on('close room', namespace='/test')
async def close(sid, message):
    await sio.emit('my response',
                   {'data': 'Room ' + message['room'] + ' is closing.'},
                   room=message['room'], namespace='/test')
    await sio.close_room(message['room'], namespace='/test')


@sio.on('my room event', namespace='/test')
async def send_room_message(sid, message):
    await sio.emit('my response', {'data': message['data']},
                   room=message['room'], namespace='/test')


@sio.on('disconnect request', namespace='/test')
async def disconnect_request(sid):
    await sio.disconnect(sid, namespace='/test')


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
    sio.start_background_task(background_task)
    initClassifier()
    web.run_app(app)

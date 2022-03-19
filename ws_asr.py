import json
from flask import Flask, render_template, request, redirect, sessions
from flask_socketio import SocketIO, emit #导入socketio包
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime

from asr_paddle import gen_wav, get_asr_text


name_space = '/asr_channel'
app = Flask(__name__)
app.secret_key = 'jzw'
socketio = SocketIO(app, async_mode='threading')
client_query = []

app.config['UPLOAD_FOLDER'] = 'upload/'

def asrThreadFunction(client_id, wav_file):
    #for i in range(10):
    #    time.sleep(1);
    #    socketio.emit("asr_message", {"text":i}, broadcast=False, namespace=name_space, room=client_id)
    
    for t, ts in gen_wav(wav_file):
        print(t, ts)
        print("INITED %s" % (datetime.now()))
        asrt=get_asr_text(t)
        res = {"text": "%s: %s\r\n" % (ts, asrt) }
        socketio.emit("asr_message", res, broadcast=False, namespace=name_space, room=client_id)
    #Stop SocketIO
    socketio.emit("disconnect", {}, broadcast=False, namespace=name_space, room=client_id)

        
        

@app.route('/uploader',methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['wavfile']
        client_id=request.form['client_id']
        print(client_id)
        print(request.files)
        wavfile=os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("%s.wav" % client_id))
        f.save(wavfile)
        print(wavfile)
        thread = socketio.start_background_task(target = asrThreadFunction, client_id=client_id, wav_file=wavfile)
    
        return 'file uploaded successfully'
    else:
        return render_template('live.html')

@socketio.on('connect', namespace=name_space)# 有客户端连接会触发该函数
def on_connect():
    # 建立连接 sid:连接对象ID
    client_id = request.sid
    client_query.append(client_id)
    emit("connect", {"sid":client_id}, broadcast=False, namespace=name_space, room=client_id)  #指定一个客户端发送消息
    #emit(event_name, broadcasted_data, broadcast=True, namespace=name_space)  #对name_space下的所有客户端发送消息
    print('有新连接,id=%s接加入, 当前连接数%d' % (client_id, len(client_query)))
    #thread = Thread(target = asrThreadFunction, args=(name_space, client_id, ))
    #thread = socketio.start_background_task(target = asrThreadFunction, client_id=client_id)
    #thread.daemon = True
    #thread.start()
    
@socketio.on('disconnect', namespace=name_space)# 有客户端断开WebSocket会触发该函数
def on_disconnect():
    # 连接对象关闭 删除对象ID
    client_query.remove(request.sid)
    print('有连接,id=%s接退出, 当前连接数%d' % (request.sid, len(client_query)))

# on('消息订阅对象', '命名空间区分')
@socketio.on('message', namespace=name_space)
def on_message(message):
    """ 服务端接收消息 """
    print('从id=%s客户端中收到消息，内容如下:' % request.sid)
    client_id = request.sid
    print(message)
    emit('my_response_message', "我收到了你的信息", broadcast=False, namespace=name_space, room=client_id)  #指定一个客户端发送消息
    # emit('my_response_message', broadcasted_data, broadcast=True, namespace=name_space)  #对name_space下的所有客户端发送消息
    
@app.route('/') #初始化页面
def a():
   return render_template("live.html")
 
if __name__ == '__main__':
    #PPSPEECH_HOME=`PWD` python ws_asr.py
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
    # app.run()

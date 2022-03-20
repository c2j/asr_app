import json
from flask import Flask, render_template, request, redirect, sessions
from flask_socketio import SocketIO, emit #导入socketio包
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime, timedelta
import wave
import audioop 

from asr_paddle import gen_wav, get_asr_text


name_space = '/asr_channel'
app = Flask(__name__)
app.secret_key = 'jzw'
socketio = SocketIO(app, async_mode='threading')
client_query = []
latest_wavfile_with_client_id = {}
app.config['UPLOAD_FOLDER'] = 'upload/'

def asrThreadFunction(client_id, wav_file, start_time=None):
    #for i in range(10):
    #    time.sleep(1);
    #    socketio.emit("asr_message", {"text":i}, broadcast=False, namespace=name_space, room=client_id)
    
    # 因为按时间上传音频的话，很可能会把最后一段音频是一句未结束的话，因此把最后一段音频记录下来
    # 并拼在下一段语音的前面
    latest_wavfile=None
    if client_id in latest_wavfile_with_client_id:
        del latest_wavfile_with_client_id[client_id]
    segments_count = 0
    for t, ts in gen_wav(wav_file, 16000):
        segments_count += 1
        latest_wavfile = t
        print(t, ts)
        print("INITED %s" % (datetime.now()))
        asrt=get_asr_text(t)
        #asrt = "test " + t
        # 将ts 加上开始的时间戳
        if start_time:
            #str(timedelta(seconds=ts))
            # start_time是在前端js中用Date.now()获得的时间戳
            st_datetime = datetime.fromtimestamp(start_time/1000)
            #print(st_datetime)
            new_datetime = str( st_datetime + timedelta(seconds=ts) )
            #print("start_time: ", start_time, 'ts: ', ts)
        else:
            new_datetime = str(timedelta(seconds=ts))
        res = {"text": "%s: %s\r\n" % (new_datetime, asrt) }
        socketio.emit("asr_message", res, broadcast=False, namespace=name_space, room=client_id)
    if segments_count > 1:
        latest_wavfile_with_client_id[client_id] = latest_wavfile
    #Stop SocketIO
    #socketio.emit("disconnect", {}, broadcast=False, namespace=name_space, room=client_id)

        
        

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
@socketio.on('upload_audio_stream', namespace=name_space)
def on_message(message):
    """ 服务端接收消息 """
    print('从id=%s客户端中收到消息，内容如下:' % request.sid)
    client_id = request.sid
    if message != None:
        print("msg len: ", len(message['data']))
        ts = datetime.now()
        startTime = message['startTime'] # 以毫秒为单位的时间戳
        print("Begin:",ts, "Time:", startTime)
        wavfile =  'tmp/%s-%s.wav' % (client_id, ts)            
        with wave.open(wavfile, 'wb') as af:              
            af.setnchannels(2)
            af.setparams((1, 2, 16000, 0, 'NONE', 'Uncompressed'))
            
            chunk_size = 4096
            sample_rate = 16000

            # 先把上次的最后一段拼上去
            if client_id in latest_wavfile_with_client_id:
                latest_wavfile = latest_wavfile_with_client_id[client_id]
                print("LATEST:", latest_wavfile)
                if latest_wavfile != None and os.path.exists(latest_wavfile):
                    with wave.open(latest_wavfile, 'rb') as rf:
                        params = rf.getparams()
                        nchannels, sampwidth, framerate, nframes = params[:4]
                        msg = rf.readframes(nframes)
                        converted = audioop.ratecv(msg, 2, 1, framerate, 16000, None)
                        # 写
                        af.writeframes(converted[0])
                        # 时间往回拨
                        startTime = startTime - (nframes / sample_rate) * 1000

            af.writeframes(message['data'])
        

        print(wavfile)
        thread = socketio.start_background_task(target = asrThreadFunction, client_id=client_id, wav_file=wavfile, start_time=startTime)
        
   
@app.route('/') #初始化页面
def a():
   return render_template("live_recorder.html")

def after_request(resp):
    #print("End:",datetime.now())
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

app.after_request(after_request)   
 
if __name__ == '__main__':
    #PPSPEECH_HOME=`PWD` python ws_asr.py
    # 预热ASR引擎
    #txt = get_asr_text("jijiji.wav")
    #print(txt)
    socketio.run(app, host='0.0.0.0', debug=True, port=5443, ssl_context=('secure/server.crt', 'secure/server.key'))
    # app.run()

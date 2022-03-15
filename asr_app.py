from flask import Flask, render_template, request, make_response
#from sockjs_flask.server import Server
#import sockjs_flask
import wave
import contextlib
import audioop 
from asr_paddle import get_asr_text, gen_wav 

#import redis
import time
import datetime
from flask_jsonpify import jsonpify
#from celery import Celery

app = Flask(__name__)
#redis_conn = redis.StrictRedis(host='localhost', db=5)
#pubsub_channel = 'task:pubsub:channel'
#celery_app = Celery('tasks', broker='redis://localhost:6379/0')
#sockjs_flask.add_endpoint(app, lambda msg, session: print(msg, session))


#@celery_app.task
#def send_text(txt):
#  print("Send: ", txt)
#  time.sleep(3)
#  return

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/upload_audio_stream', methods=['POST'])
def upload_audio_stream_backup(): 
    print("Begin:",datetime.datetime.now())
    cr_record_file =  'audioData.wav'             
    with wave.open(cr_record_file, 'wb') as af:              
        af.setnchannels(2)
        af.setparams((1, 2, 16000, 0, 'NONE', 'Uncompressed'))
        
        chunk_size = 4096
        sample_rate = 16000
        while True:
            chunk = request.stream.read(chunk_size)
            #print(chunk)
            if len(chunk) == 0:
              break
            #converted = audioop.ratecv(chunk, 2, 1, sample_rate, 16000, None)
            #af.writeframes(converted[0])
            af.writeframes(chunk)
    

    txt = ''
    #try:
    txt = get_asr_text(cr_record_file)
    
    #except:
    #    pass
    json_data = {  'text': txt   
            }
          
    print(json_data)
    #TODO 然后将文本放到消息队列中
    #send_text(dial)
    #redis_conn.publish(pubsub_channel, str(json_data))

    
    return jsonpify(json_data, ensure_ascii=False)  

@app.route('/upload_audio_stream_kedaxunfei', methods=['POST'])
def upload_audio_stream(): 
    print("Begin:",datetime.datetime.now())

    def gen_stream(stream):
        chunk_size = 16000
        sample_rate = 16000
        audioStatus = 0
        while True:        
            if audioStatus == 0:
                audioStatus = 1                
            wavbyte =stream.read(chunk_size)
            #print(wavbyte)
            if len(wavbyte) == 0:
                audioStatus = 4          
            yield (audioStatus, wavbyte)    
            if audioStatus == 4:
                break
            elif audioStatus == 1:
                audioStatus = 2

    streams = gen_stream(request.stream)          
    txt = None
    for txt in gen_asr_text( streams, online=True ):
        #print(txt)
        pass
    
    
    #except:
    #    pass
    json_data = {  'text': txt   
            }
          
    print(json_data)
    #TODO 然后将文本放到消息队列中
    #redis_conn.publish(pubsub_channel, str(json_data))

    
    return jsonpify(json_data, ensure_ascii=False)  

def after_request(resp):
    print("End:",datetime.datetime.now())
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

app.after_request(after_request)

#@app.route('/sockjs-node/info')
#def sockjs_info():
#  resp = make_response("OK")
#  return resp    

if __name__ == '__main__':
    #
    txt = get_asr_text("jijiji.wav")
    print(txt)
    #避免对json返回的中文进行编码
    app.config['JSON_AS_ASCII'] = False
    app.config.update(RESTFUL_JSON=dict(ensure_ascii = False))
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5443, ssl_context=('secure/server.crt', 'secure/server.key'))
    #server = Server(('0.0.0.0', 5002), app.wsgi_app)
    #server.server_forever()


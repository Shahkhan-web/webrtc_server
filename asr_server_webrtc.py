import json 
import os
import concurrent.futures
import asyncio
from aiortc import RTCSessionDescription, RTCPeerConnection
from av.audio.resampler import AudioResampler
from av.video import VideoFrame
from pathlib import Path
from vosk import KaldiRecognizer, Model  
from quart import Quart, request, render_template
from quart_cors import cors, route_cors 
import cv2
import base64
import numpy as np 
 

vosk_interface = os.environ.get('VOSK_SERVER_INTERFACE', '0.0.0.0')
vosk_port = int(os.environ.get('VOSK_SERVER_PORT', 2700))
vosk_model_path = os.environ.get('VOSK_MODEL_PATH', 'model')
vosk_cert_file = os.environ.get('VOSK_CERT_FILE', None)
vosk_key_file = os.environ.get('VOSK_KEY_FILE', None)
vosk_dump_file = os.environ.get('VOSK_DUMP_FILE', None)

model = Model(vosk_model_path)
pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))
dump_fd = None if vosk_dump_file is None else open(vosk_dump_file, "wb")
 
 
app = Quart(__name__, template_folder='static') 
app = cors(app, allow_origin="6z98ptwz-8080.inc1.devtunnels.ms", allow_credentials=True)

def process_chunk(rec, message):    
    try:
        res = rec.AcceptWaveform(message)
    except Exception:
        result = None
    else:   
        if res > 0:
            result = rec.Result()
        else:
            result = rec.PartialResult()
    return result

class KaldiTask:
    def __init__(self, user_connection):
        self.__resampler = AudioResampler(format='s16', layout='mono', rate=48000)
        self.__pc = user_connection
        self.__audio_task = None
        self.__track = None
        self.__channel = None
        self.__recognizer = KaldiRecognizer(model, 48000)

    async def set_audio_track(self, track):
        self.__track = track

    async def set_text_channel(self, channel):
        self.__channel = channel

    async def start(self):
        self.__audio_task = asyncio.create_task(self.__run_audio_xfer())

    async def stop(self):
        if self.__audio_task is not None:
            self.__audio_task.cancel()
            self.__audio_task = None

    async def __run_audio_xfer(self):
        loop = asyncio.get_running_loop()

        max_frames = 20
        frames = []
        while True:
            fr = await self.__track.recv()
            frames.append(fr)

            # We need to collect frames so we don't send partial results too often
            if len(frames) < max_frames:
                continue

            dataframes = bytearray(b'')
            for fr in frames:
                for rfr in self.__resampler.resample(fr):
                    dataframes += bytes(rfr.planes[0])[:rfr.samples * 2]
            frames.clear()

            if dump_fd != None:
                dump_fd.write(bytes(dataframes))

            result = await loop.run_in_executor(pool, process_chunk, self.__recognizer, bytes(dataframes))
            print(result)
            self.__channel.send(result)
 
async def save_video_as_jpg(track):
    frame = await track.recv()
    img = frame.to_ndarray(format='bgr24')
    print('saving frame...')
    cv2.imwrite('frame.jpg', img)
    return

@app.route('/offer', methods=['POST'])
@route_cors(allow_origin="*")  # This will enable CORS for this specific route
async def offer():
    #get the sdp and type from the request which is coming as stringified json
    params = await request.get_json()
    print('connecting through sdp...')
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])

    pc = RTCPeerConnection()

    kaldi = KaldiTask(pc)

    @pc.on('datachannel')
    async def on_datachannel(channel):
        channel.send('{}') # Dummy message to make the UI change to "Listening"
        await kaldi.set_text_channel(channel)
        await kaldi.start()

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == 'failed':
            await pc.close()

    @pc.on('track')
    async def on_track(track):
        if track.kind == 'audio':
            await kaldi.set_audio_track(track)

        if track.kind == 'video':
            while True:
                await save_video_as_jpg(track);
        
        @track.on('ended')
        async def on_ended():
            await kaldi.stop() 

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return json.dumps({
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    })


if __name__ == '__main__':
    # Start the dev server
    app.run(host=vosk_interface, port=vosk_port, debug=True)
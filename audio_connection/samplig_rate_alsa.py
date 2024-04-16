import sounddevice as sd

S = sd.InputStream()
print('fs = ', S.samplerate)
print('blocksize = ', S.blocksize)
print('channels = ', S.channels)
print('latency = ', S.latency)
print('devinfo = ', S.device[0])
print(sd.query_devices())
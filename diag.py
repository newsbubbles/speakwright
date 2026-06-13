import io, json, struct, urllib.request
import numpy as np
import soundfile as sf

text = ("This is a complete sentence that should produce several seconds of audio "
        "and it must not get cut off early during playback in the browser.")
req = urllib.request.Request(
    "http://127.0.0.1:8765/v1/audio/speech",
    data=json.dumps({"input": text, "response_format": "wav"}).encode(),
    headers={"Content-Type": "application/json"},
)
content = urllib.request.urlopen(req, timeout=120).read()
print("bytes:", len(content))

# WAV header sanity
riff = content[0:4]
riff_size = struct.unpack("<I", content[4:8])[0]
wave = content[8:12]
print("RIFF tag:", riff, "WAVE tag:", wave)
print("riff_size field:", riff_size, "| expected (len-8):", len(content) - 8,
      "| match:", riff_size == len(content) - 8)

# find data chunk size
i = 12
data_size = None
while i + 8 <= len(content):
    cid = content[i:i+4]
    csz = struct.unpack("<I", content[i+4:i+8])[0]
    if cid == b"data":
        data_size = csz
        print("data chunk size field:", csz, "| bytes actually present after header:", len(content) - (i+8),
              "| match:", csz == len(content) - (i+8))
        break
    i += 8 + csz + (csz & 1)

# decode + sanity
data, sr = sf.read(io.BytesIO(content))
print("samplerate:", sr, "samples:", len(data), "duration_s:", round(len(data)/sr, 2))
print("dtype:", data.dtype, "nan:", bool(np.isnan(data).any()), "inf:", bool(np.isinf(data).any()),
      "peak:", float(np.max(np.abs(data))) if len(data) else 0.0)

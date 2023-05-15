#%%
import youtube_dl
import whisper

# Download audio from YouTube video
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'outtmpl':'audio.wav'
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=uzy-KAIgAx8'])

# Load Whisper model
model = whisper.load_model("base")

# Transcribe audio file
result = model.transcribe("./audio.wav")
print(result["text"])
# %%
from transformers import HfAgent
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

from PIL import Image
x = Image.open('test.jpg')

agent.run("here is a image named image, what does it belong to?", image=x, remote=True)
# %%
!pip install transformers -U -q
# %%
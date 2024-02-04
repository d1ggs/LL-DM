import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", device_map="cuda")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large", device_map="cuda")

# from text
text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt").to("cuda")
audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()

# from audio
audio, orig_freq = torchaudio.load("https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav")
audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt").to("cuda")
audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()

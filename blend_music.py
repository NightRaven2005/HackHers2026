from audiocraft.models import MusicGen
from pydub import AudioSegment
import os
import torch
import scipy.io.wavfile as wavfile

device = "cpu"
print(f"Using device: {device}")

model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
model.set_generation_params(duration=30)


def build_prompt(genreA, genreB, alpha):
    weightA = round(alpha * 100)
    weightB = round((1 - alpha) * 100)

    # Determine dominant style
    if alpha > 0.5:
        dominance = f"Strong emphasis on {genreA}."
    elif alpha < 0.5:
        dominance = f"Strong emphasis on {genreB}."
    else:
        dominance = "Balanced fusion of both styles."

    prompt = (
        f"A music track that is {weightA}% {genreA} and {weightB}% {genreB}. "
        f"{dominance} "
        f"High quality studio production. Rich instrumentation. "
        f"Clear structure and dynamic composition."
    )

    return prompt


def generate_from_blend(genreA: str, genreB: str, alpha: float, out_dir: str) -> str:
    prompt = build_prompt(genreA, genreB, alpha)
    print("Prompt:", prompt)

    wav = model.generate([prompt])

    audio = wav[0].cpu().numpy().T  # transpose
    sampling_rate = 32000  # MusicGen default

    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{genreA}_{genreB}_{int(alpha * 100)}.wav")
    wavfile.write(wav_path, sampling_rate, audio)

    mp3_path = wav_path.replace(".wav", ".mp3")
    AudioSegment.from_wav(wav_path).set_frame_rate(44100).export(mp3_path, format="mp3")
    os.remove(wav_path)

    return mp3_path


# TEST
generate_from_blend("classical", "hip hop", 0.7, "music_output")
import os
import time
import wave
import numpy as np

def generate_from_blend(genreA, genreB, alpha, out_dir):
    fname = f"gen_{int(time.time())}.wav"
    out_path = os.path.join(out_dir, fname)

    # Generate 2 seconds of random noise
    sample_rate = 22050
    duration = 2
    samples = np.random.uniform(-1, 1, sample_rate * duration)

    with wave.open(out_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())

    return out_path
def predict_genre(audio_path):
    import random
    base = random.random()
    return {
        "classical": round(base, 2),
        "hiphop": round(1 - base, 2)
    }
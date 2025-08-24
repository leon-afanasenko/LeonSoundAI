import gradio as gr
from TTS.api import TTS      # pip install TTS
from audiocraft.models import MusicGen
from pydub import AudioSegment

# Предполагается что у вас уже есть voice embedding или fine-tuned модель!
VOICE_MODEL_PATH = "path/to/your/tts_model_or_voice_embedding"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
tts.load_voice(VOICE_MODEL_PATH)

music_model = MusicGen.get_pretrained("facebook/musicgen-small")

def generate_song_with_voice(lyrics, genre, duration, progress=gr.Progress(track_tqdm=True)):
    # 1. Генерируем вокал (ваш голос) по тексту
    tts_kwargs = {
        "text": lyrics,
        "speaker_wav": VOICE_MODEL_PATH,  # если ваша модель принимает embedding, иначе уберите
        "language": "ru",                # или другой язык
    }
    vocal_path = "Leon_vibe/vocal.wav"
    tts.tts_to_file(**tts_kwargs, file_path=vocal_path)
    
    # 2. Генерируем минус (музыку) под жанр
    prompt = f"{genre} instrumental"
    music_model.set_generation_params(duration=int(duration))
    music = music_model.generate([prompt])
    music_path = "Leon_vibe/music.wav"
    audio_write(music_path, music[0].cpu(), music_model.sample_rate)

    # 3. Микшируем вокал и минус
    vocal = AudioSegment.from_wav(vocal_path)
    instrumental = AudioSegment.from_wav(music_path)
    # Можно выровнять длину/громкость и т.д.
    out = instrumental.overlay(vocal)
    out_path = "Leon_vibe/final_song.wav"
    out.export(out_path, format="wav")
    return out_path

with gr.Blocks() as demo:
    with gr.Tab("Песня вашим голосом"):
        gr.Markdown("## Сгенерируйте песню своим голосом\n1. Введите текст песни\n2. Выберите жанр\n3. Получите песню!")
        lyrics_input = gr.Textbox(label="Текст песни")
        genre_input = gr.Dropdown(["pop", "rock", "rap", "rnb", "jazz", "lofi", "electronic"], label="Жанр", value="pop")
        duration_input = gr.Slider(minimum=10, maximum=60, value=30, step=1, label="Длительность (секунд)")
        generate_song_btn = gr.Button("Сгенерировать")
        audio_output = gr.Audio(label="Результат", type="filepath")
        generate_song_btn.click(
            generate_song_with_voice,
            inputs=[lyrics_input, genre_input, duration_input],
            outputs=audio_output
        )

# demo.launch() # Не забудьте добавить/объединить с вашим основным UI
import os
import sys
import time
import subprocess
from pathlib import Path
import gradio as gr
from pydub import AudioSegment
import numpy as np
import torch
from colorama import init, Fore, Style
from audiocraft.models import MusicGen
from TTS.api import TTS

# ==== ИНИЦИАЛИЗАЦИЯ ====
init(autoreset=True)
os.environ["COQUI_TOS_AGREED"] = "1"

def log(msg, color=Fore.RESET, end="\n"):
    print(color + msg + Style.RESET_ALL, end=end)

log("====== ЗАПУСК Leon's Vibe Creator ======", Fore.MAGENTA)
log("Папка для музыки: Leon_vibe", Fore.YELLOW)
log("Папка для голоса: Leon_voice", Fore.YELLOW)
log("Ожидание запуска моделей...", Fore.YELLOW)

OUTPUT_DIR = Path("Leon_vibe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR = Path("Leon_voice")
VOICE_DIR.mkdir(parents=True, exist_ok=True)

# ==== МОДЕЛИ ====
start_time = time.time()
try:
    log("Загружаю модель MusicGen (facebook/musicgen-small)...", Fore.YELLOW)
    musicgen = MusicGen.get_pretrained("facebook/musicgen-small")
    log("Модель MusicGen загружена. Готово.", Fore.GREEN)
except Exception as e:
    log(f"ОШИБКА MusicGen: {e}", Fore.RED)
    sys.exit(1)

try:
    log("Загружаю TTS XTTS v2...", Fore.YELLOW)
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    log("TTS XTTS v2 загружен.", Fore.GREEN)
except Exception as e:
    log(f"ОШИБКА TTS: {e}", Fore.RED)
    sys.exit(1)

total_load = time.time() - start_time
log(f"Все модели загружены за {total_load:.1f} сек. Можно начинать!", Fore.GREEN)

# ==== ПОДДЕРЖКА (старые функции) ====
def create_safe_filename(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "_- ").rstrip()
    return safe or "track"

def audio_write(path: str, audio_tensor: torch.Tensor, sample_rate: int):
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim > 1: audio_np = audio_np[0]
    audio_int16 = (audio_np * 32767).astype(np.int16)
    AudioSegment(audio_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1).export(path, format="wav")

def list_audio_files():
    files = list(OUTPUT_DIR.glob("*.wav")) + list(OUTPUT_DIR.glob("*.mp3"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [str(p.resolve()) for p in files]

def list_voice_files():
    files = list(VOICE_DIR.glob("*.wav"))
    return [str(f) for f in files]

def delete_file(path_str: str):
    try:
        if path_str and Path(path_str).exists():
            Path(path_str).unlink()
        updated_choices = gr.update(choices=list_audio_files())
        return updated_choices, updated_choices
    except Exception as e:
        log(f"Ошибка при удалении файла: {e}", Fore.RED)
        return gr.update(choices=list_audio_files()), gr.update(choices=list_audio_files())

# Импортируй свои функции из ffmpeg_utils, если нужно:
try:
    from ffmpeg_utils import print_audio_comparison, process_existing_audio
except ImportError:
    def process_existing_audio(x, *args, **kwargs):
        return x, "ffmpeg_utils не найден, обработка не производится"

# ==== MusicGen Генерация ====
def generate_music_workflow(prompt: str, duration: int, track_name: str, process_audio: bool, progress=gr.Progress(track_tqdm=True)):
    start = time.time()
    try:
        musicgen.set_generation_params(duration=int(duration))
        num_steps = 30
        for i in progress.tqdm(range(num_steps), desc="Шаг 1/2: Генерация музыки..."):
            time.sleep(max(0.1, duration / num_steps / 2))  # чтобы всегда была анимация
        wavs = musicgen.generate([prompt])
        progress(0.9, desc="Шаг 2/2: Сохранение и обработка...")
        safe_name = create_safe_filename(track_name)
        wav_path = OUTPUT_DIR / f"{safe_name}.wav"
        audio_write(str(wav_path), wavs[0].cpu(), musicgen.sample_rate)
        if process_audio:
            processed_path, _ = process_existing_audio(str(wav_path))
            result = processed_path or str(wav_path)
        else:
            result = str(wav_path)
        elapsed = time.time() - start
        log(f"[MusicGen] Трек '{track_name}' создан за {elapsed:.1f} сек.", Fore.CYAN)
        return result
    except Exception as e:
        log(f"[MusicGen] Ошибка: {e}", Fore.RED)
        raise gr.Error(f"Критическая ошибка: {e}")

# ==== TTS Voice Cloning (new) ====
def save_uploaded_voice(voice_file):
    if not voice_file:
        raise gr.Error("Сначала выберите .wav файл для загрузки!")
    if isinstance(voice_file, tuple):  # Gradio >=4
        voice_file = voice_file[0]
    if not os.path.isfile(voice_file):
        raise gr.Error("Файл не найден или не был загружен.")
    fname = VOICE_DIR / Path(voice_file).name
    try:
        os.rename(voice_file, fname)
    except Exception as e:
        log(f"Ошибка при сохранении голоса: {e}", Fore.RED)
        raise gr.Error(f"Ошибка при сохранении: {e}")
    log(f"Загружен новый голос: {fname.name}", Fore.GREEN)
    return str(fname)

def generate_song_with_voice(lyrics, genre, duration, voice_sample_path, progress=gr.Progress(track_tqdm=True)):
    if not voice_sample_path or not os.path.isfile(voice_sample_path):
        raise gr.Error("Выберите голосовой файл для генерации!")
    t0 = time.time()
    try:
        # 1. Генерируем вокал
        progress(0.05, desc="Генерация вокала (TTS)...")
        vocal_path = OUTPUT_DIR / "vocal.wav"
        tts.tts_to_file(
            text=lyrics,
            speaker_wav=voice_sample_path,
            language="ru",
            file_path=str(vocal_path),
        )
        # 2. Генерируем музыку под жанр
        progress(0.5, desc="Генерация музыки (MusicGen)...")
        musicgen.set_generation_params(duration=int(duration))
        prompt = f"{genre} instrumental"
        music = musicgen.generate([prompt])
        music_path = OUTPUT_DIR / "music.wav"
        audio_np = music[0].cpu().numpy()
        if audio_np.ndim > 1: audio_np = audio_np[0]
        audio_int16 = (audio_np * 32767).astype(np.int16)
        AudioSegment(
            audio_int16.tobytes(),
            frame_rate=musicgen.sample_rate,
            sample_width=2,
            channels=1
        ).export(music_path, format="wav")
        # 3. Микшируем вокал и минус
        progress(0.8, desc="Микширование вокала и минуса...")
        vocal = AudioSegment.from_wav(vocal_path)
        instrumental = AudioSegment.from_wav(music_path)
        min_len = min(len(vocal), len(instrumental))
        out = instrumental[:min_len].overlay(vocal[:min_len])
        out_path = OUTPUT_DIR / "final_song.wav"
        out.export(out_path, format="wav")
        elapsed = time.time() - t0
        log(f"[TTS+MusicGen] Песня с вашим голосом готова за {elapsed:.1f} сек.", Fore.CYAN)
        return str(out_path)
    except Exception as e:
        log(f"[TTS+MusicGen] Ошибка: {e}", Fore.RED)
        raise gr.Error(f"Ошибка генерации песни: {e}")

# ==== UI ====
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.blue)) as demo:
    gr.Markdown("# 🎵 Leon's Vibe Creator (MusicGen + XTTS v2) 🎵")
    
    with gr.Tab("Создание и Улучшение"):
        gr.Markdown("### Шаг 1: Создайте новый трек")
        with gr.Row():
            prompt_input = gr.Textbox(label="Введите промпт (лучше на англ.)", lines=2, value="lofi relaxing piano")
            with gr.Column():
                track_name_input = gr.Textbox(label="Название трека (имя файла)", value="Leon_music")
                duration_input = gr.Slider(minimum=5, maximum=60, value=20, step=1, label="Длительность (секунд)")
        process_checkbox = gr.Checkbox(label="Сразу улучшить аудио (ffmpeg)", value=True)
        generate_button = gr.Button("1. Сгенерировать", variant="primary")
        generated_audio_output = gr.Audio(label="Результат генерации", type="filepath")

        gr.Markdown("--- \n ### Шаг 2: Улучшите существующий трек")
        files_list_process = gr.Dropdown(label="Выберите файл для улучшения", choices=list_audio_files(), interactive=True)
        process_button = gr.Button("2. Улучшить выбранный файл")
        processed_audio_output = gr.Audio(label="Результат улучшения", type="filepath")
        status_text = gr.Textbox(label="Статус", interactive=False)

    with gr.Tab("Файловый менеджер"):
        files_list_manage = gr.Dropdown(label="Список файлов в папке", choices=list_audio_files(), interactive=True)
        with gr.Row():
            play_button = gr.Button("Прослушать выбранный файл")
            delete_button = gr.Button("Удалить выбранный файл")
        audio_player = gr.Audio(label="Плеер", type="filepath")

    with gr.Tab("Пой как Леон! (Voice Song)"):
        gr.Markdown("## 1. Загрузите ваш чистый голос (.wav, ~10 секунд, без музыки):")
        with gr.Row():
            voice_upload = gr.Audio(label="Файл вашего голоса", type="filepath")
            upload_btn = gr.Button("Загрузить голос")
        voice_files_dropdown = gr.Dropdown(label="Выберите голос для генерации", choices=list_voice_files())

        gr.Markdown("## 2. Введите текст, выберите жанр и длительность:")
        lyrics_input = gr.Textbox(label="Текст песни")
        genre_input = gr.Dropdown(["pop", "rock", "rap", "jazz", "lofi", "electronic"], label="Жанр", value="pop")
        duration_input2 = gr.Slider(minimum=10, maximum=60, value=30, step=1, label="Длительность (сек)")
        generate_song_btn = gr.Button("Сгенерировать песню")
        song_output = gr.Audio(label="Ваша песня", type="filepath")

        def on_upload(voice_file):
            # Если пустой - ошибка
            if not voice_file:
                raise gr.Error("Сначала выберите .wav файл для загрузки!")
            path = save_uploaded_voice(voice_file)
            return gr.update(choices=list_voice_files(), value=path)
        upload_btn.click(on_upload, inputs=voice_upload, outputs=voice_files_dropdown)
        generate_song_btn.click(
            generate_song_with_voice,
            inputs=[lyrics_input, genre_input, duration_input2, voice_files_dropdown],
            outputs=song_output,
        )

    # ==== КНОПКИ (старый интерфейс) ====
    def on_generate_and_update(prompt, duration, track_name, process_audio, progress=gr.Progress(track_tqdm=True)):
        return generate_music_workflow(prompt, duration, track_name, process_audio, progress), \
            gr.update(choices=list_audio_files()), gr.update(choices=list_audio_files())

    def on_process_and_update(filepath, progress=gr.Progress(track_tqdm=True)):
        path, status = process_existing_audio(filepath, progress)
        return path, status, gr.update(choices=list_audio_files()), gr.update(choices=list_audio_files())

    generate_button.click(
        on_generate_and_update,
        inputs=[prompt_input, duration_input, track_name_input, process_checkbox],
        outputs=[generated_audio_output, files_list_process, files_list_manage]
    )
    process_button.click(
        on_process_and_update,
        inputs=[files_list_process],
        outputs=[processed_audio_output, status_text, files_list_process, files_list_manage]
    )
    play_button.click(lambda p: p, inputs=[files_list_manage], outputs=[audio_player])
    delete_button.click(delete_file, inputs=[files_list_manage], outputs=[files_list_process, files_list_manage])

if __name__ == "__main__":
    log("===> Интерфейс загружен, можно заходить по адресу ниже", Fore.GREEN)
    demo.launch()
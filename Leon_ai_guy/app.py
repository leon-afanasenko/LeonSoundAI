import os
import sys
import subprocess
from pathlib import Path
import gradio as gr
from pydub import AudioSegment
import numpy as np
import torch
import time
from colorama import init, Fore, Style
from audiocraft.models import MusicGen
from ffmpeg_utils import print_audio_comparison, process_existing_audio

# --- Инициализация и настройки ---
init(autoreset=True)
OUTPUT_DIR = Path("Leon_vibe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(Fore.YELLOW + "Запуск приложения... Все треки будут в папке 'Leon_vibe'")

# --- Загрузка модели ---
print(Fore.YELLOW + "Загружаю модель MusicGen (facebook/musicgen-small)...")
model = MusicGen.get_pretrained("facebook/musicgen-small")
print(Fore.GREEN + "Модель MusicGen загружена. Готово.")

# --- Функции-помощники ---
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

def delete_file(path_str: str):
    if path_str and Path(path_str).exists():
        Path(path_str).unlink()
    updated_choices = gr.update(choices=list_audio_files())
    return updated_choices, updated_choices

# --- Основной workflow генерации с КРАСИВОЙ АНИМАЦИЕЙ ---
def generate_music_workflow(prompt: str, duration: int, track_name: str, process_audio: bool, progress=gr.Progress(track_tqdm=True)):
    try:
        # --- Шаг 1: Генерация с плавной анимацией ---
        model.set_generation_params(duration=int(duration))
        
        # Визуальная симуляция прогресса для лучшего UX
        num_steps = 30 # Количество "шагов" анимации
        for _ in progress.tqdm(range(num_steps), desc="Шаг 1/2: Генерация музыки..."):
            # Время ожидания зависит от длительности трека, чтобы анимация была реалистичной
            time.sleep(duration / num_steps)
        
        # Настоящая генерация
        wavs = model.generate([prompt])
        
        # --- Шаг 2: Сохранение и Обработка ---
        progress(0.9, desc="Шаг 2/2: Сохранение и обработка...")
        safe_name = create_safe_filename(track_name)
        wav_path = OUTPUT_DIR / f"{safe_name}.wav"
        audio_write(str(wav_path), wavs[0].cpu(), model.sample_rate)
        
        if process_audio:
            processed_path, _ = process_existing_audio(str(wav_path))
            return processed_path or str(wav_path)
        else:
            return str(wav_path)
            
    except Exception as e:
        raise gr.Error(f"Критическая ошибка: {e}")

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.blue)) as demo:
    gr.Markdown("# 🎵 Leon's Vibe Creator (MusicGen) 🎵")
    
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

    # --- Логика кнопок ---
    def on_generate_and_update(prompt, duration, track_name, process_audio, progress=gr.Progress(track_tqdm=True)):
        path = generate_music_workflow(prompt, duration, track_name, process_audio, progress)
        updated_list = gr.update(choices=list_audio_files(), value=path)
        return path, updated_list, updated_list

    def on_process_and_update(filepath, progress=gr.Progress(track_tqdm=True)):
        path, status = process_existing_audio(filepath, progress)
        updated_list = gr.update(choices=list_audio_files(), value=path)
        return path, status, updated_list, updated_list
    
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
    demo.launch()
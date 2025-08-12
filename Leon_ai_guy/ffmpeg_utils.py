import subprocess
from colorama import init, Fore, Style
from pathlib import Path
import sys

# --- Инициализация colorama ---
init(autoreset=True)

FFPROBE_PATH = "ffprobe"

def get_audio_info(filename_str: str):
    """
    Получает детальную информацию об аудиофайле с помощью ffprobe.
    """
    filename = Path(filename_str)
    if not filename.exists(): return None
    
    cmd = [
        FFPROBE_PATH, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,bit_rate,codec_name",
        "-show_entries", "format=duration,size",
        "-of", "default=noprint_wrappers=1:nokey=1", str(filename)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0: return None
    
    try:
        lines = result.stdout.strip().split('\n')
        return {
            "path": filename,
            "codec": lines[0], "sample_rate": int(lines[1]),
            "channels": int(lines[2]), "bit_rate": int(lines[3]) if lines[3].isdigit() else 0,
            "duration": float(lines[4]), "size": int(lines[5])
        }
    except Exception as e:
        print(Fore.RED + f"Ошибка парсинга данных ffprobe: {e}")
        return None

def format_size(size_bytes):
    """Форматирует байты в КБ или МБ для красивого вывода."""
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def print_audio_comparison(orig_info, proc_info, gain_db):
    """
    Печатает красивый и детальный отчет "ДО и ПОСЛЕ".
    """
    if not orig_info or not proc_info:
        print(Fore.YELLOW + "Не удалось получить информацию для сравнения файлов.")
        return
        
    print(Style.BRIGHT + Fore.CYAN + "\n" + "="*50)
    print(" " * 15 + "СТУДИЙНЫЙ ОТЧЕТ")
    print("="*50)
    
    print(Fore.WHITE + "\nДО ОБРАБОТКИ (Исходный файл):")
    print(f"  Файл:      {orig_info['path'].name}")
    print(f"  Формат:    {orig_info['codec'].upper()} (WAV)")
    print(f"  Размер:    {format_size(orig_info['size'])}")
    print(f"  Битрейт:   {orig_info['bit_rate'] // 1000} kbps")
    print(f"  Длительность: {orig_info['duration']:.2f} сек")
    
    print(Fore.GREEN + "\nПОСЛЕ ОБРАБОТКИ (Улучшенный файл):")
    print(f"  Файл:      {proc_info['path'].name}")
    print(f"  Формат:    {proc_info['codec'].upper()} (MP3)")
    print(f"  Размер:    {format_size(proc_info['size'])}")
    print(f"  Битрейт:   {proc_info['bit_rate'] // 1000} kbps")
    print(f"  Длительность: {proc_info['duration']:.2f} сек")
    
    print(Fore.YELLOW + "\nЧто было сделано:")
    print(f"  - Применена динамическая нормализация (dynaudnorm)")
    print(f"  - Громкость увеличена на +{gain_db} dB")
    print(f"  - Файл сжат в MP3 с высоким битрейтом (320k)")
    print(Style.BRIGHT + Fore.CYAN + "="*50 + "\n")


def process_existing_audio(input_path_str: str, progress=None):
    """
    Главная функция: улучшает трек и ВЫВОДИТ ДЕТАЛЬНЫЙ ОТЧЕТ.
    """
    if not input_path_str: return None, "Файл не выбран."
    input_path = Path(input_path_str)
    if not input_path.exists():
        return None, f"Ошибка: Файл не найден: {input_path.name}"
    
    if progress: progress(0.2, desc=f"Улучшаю: {input_path.name}...")
    output_path = input_path.with_name(f"{input_path.stem}_ENHANCED.mp3")
    gain_db = 6 # Громкость

    print(Fore.CYAN + f"-> FFmpeg: улучшаю {input_path.name} -> {output_path.name}")
    try:
        filter_chain = f"dynaudnorm=f=200:g=15,volume={gain_db}dB"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(input_path), "-af", filter_chain, "-c:a", "libmp3lame", "-b:a", "320k", str(output_path), "-y"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        if progress: progress(1, desc="Готово!")
        
        # --- ВЫВОДИМ ОТЧЕТ СРАЗУ ПОСЛЕ ОБРАБОТКИ ---
        orig_info = get_audio_info(str(input_path))
        proc_info = get_audio_info(str(output_path))
        print_audio_comparison(orig_info, proc_info, gain_db)
        
        return str(output_path.resolve()), "Трек успешно улучшен!"
    except subprocess.CalledProcessError as e:
        error_msg = f"-> FFmpeg ERROR:\n{e.stderr or e.stdout}"
        print(Fore.RED + error_msg, file=sys.stderr)
        return None, error_msg
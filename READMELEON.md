# 🎵 Leon's Vibe Creator (Supercharged AudioCraft)

This is a heavily enhanced and modified version of Meta's AudioCraft project, transformed into a full-featured music production studio with a powerful, user-friendly web interface.

![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

## 🔥 Key Enhancements by Leon

This project has been fundamentally reworked from a simple script into a complete production tool:

- **🎹 Full-Featured Web Studio:** A comprehensive Gradio UI replaces the command line. It features separate tabs for music generation and file management, giving you full control over your workflow.
- **🔊 Advanced Audio Enhancement:** Automatically process newly generated tracks or enhance existing ones with a powerful FFMPEG filter chain (`dynaudnorm`, volume boost) to make them sound full and professional.
- **🎛️ On-Demand Processing:** Don't like the raw WAV? Select any track from your library and run the enhancement process on-demand with a single click.
- **📂 Complete File Management:** Manage your track library directly from the UI. The integrated file manager lets you browse, play, and delete any track in your collection.
- **🏷️ Custom Naming & Organization:** Give your tracks custom names directly in the UI. All files are automatically saved into a dedicated `Leon_vibe` folder.
- **📊 Real-Time Feedback:** Never guess if the app is working. An interactive progress bar provides clear, step-by-step feedback during the entire generation and processing workflow.
- **📈 Console Analytics:** After each enhancement, see a direct before-and-after comparison of the track's audio properties (bitrate, duration) printed in the console.

## 🚀 Application Features

The UI is organized into two main tabs for a clean workflow:

#### 1. Creation & Enhancement Tab

- **Generate & Enhance:** The main workflow. Write a prompt, name your track, select a duration, and check the "Enhance" box. The app will generate a raw WAV file and immediately process it into a high-quality, loud MP3.
- **Enhance Existing Track:** Use the dropdown to select any track from your `Leon_vibe` folder and click "Enhance" to apply the FFMPEG post-processing.

#### 2. File Manager Tab

- **Browse Your Library:** The dropdown shows all `.wav` and `.mp3` files in your collection.
- **Listen & Delete:** Select a file and use the "Play" button to listen to it in the embedded player, or the "Delete" button to permanently remove it.

## ⚙️ Setup & Installation

This project requires **Python 3.11** and uses the **Conda** environment manager.

1.  **Create the Environment:**

    ```shell
    conda create --name audiocraft_final -c conda-forge python=3.11 -y
    conda activate audiocraft_final
    ```

2.  **Install PyTorch:**

    ```shell
    conda install pytorch torchvision torchaudio -c pytorch -c conda-forge -y
    ```

3.  **Install Dependencies:**
    ```shell
    # Run from the root 'audiocraft' directory
    pip install -e .
    # Install additional libraries for the UI and processing
    pip install ffmpeg-python colorama pydub
    ```

## 🚀 Quick Start

1.  **Activate the Conda environment:**

    ```shell
    conda activate audiocraft_final
    ```

2.  **Navigate to the project directory:**

    ```shell
    cd /path/to/your/project/audiocraft/Leon_ai_guy/
    ```

3.  **Launch the application:**

    ```shell
    python3 app.py
    ```

4.  **Create:** Open the local URL (e.g., `http://127.0.0.1:7860`) in your browser and start creating!

## 📜 License

- The original code in this repository is released under the MIT license, as found in the [LICENSE file](LICENSE).
  **Copyright (c) Meta Platforms, Inc. and affiliates, and Leon.**
- The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).

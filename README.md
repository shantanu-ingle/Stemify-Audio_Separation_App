# Stemify: AI-Powered Multi-Instrument and Vocal Separation
## Overview
Stemify is an AI-driven application designed to separate mixed audio into individual stems—vocals, drums, bass, and other instruments—with minimal artifacts. Leveraging a phase-aware DeepMultiSourceUNet model trained on the MUSDB18 dataset, Stemify achieves high-quality separation suitable for music production, remixing, education, and DJing. A Flask-based web interface allows users to upload WAV/MP3 files, process them, and download isolated stems.

## Features
AI Stem Separation: Isolates vocals, drums, bass, and other instruments using a phase-aware U-Net model.
High Fidelity: Minimizes artifacts with a composite loss function (MSE, phase consistency, independence, magnitude sum).
Web Interface: Flask-based UI supports WAV/MP3 uploads, stem selection, and downloadable outputs.
Tempo & Key Detection: Planned module for BPM and key analysis to aid remixing.
Efficient Processing: Runs on modest hardware (< 4 GB vRAM).

## Output
![Screenshot 2025-05-24 085102](https://github.com/user-attachments/assets/e536c9d2-c699-4ee6-bf6d-ea2f6009c4d3)
![Screenshot 2025-05-24 090357](https://github.com/user-attachments/assets/36f6d111-8db1-474f-a202-0e8b8bfddf80)
![Screenshot 2025-05-24 090414](https://github.com/user-attachments/assets/956e5f11-05c1-4ffa-9570-99eacbf50ffa)
![Screenshot 2025-05-24 090427](https://github.com/user-attachments/assets/ab9865a9-0e3b-4771-add9-8e2c5c766c6d)
![Screenshot 2025-05-24 090450](https://github.com/user-attachments/assets/d83a9de0-a995-4bb8-83f7-e1e5bbf45285)





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
![1736697169969](https://github.com/user-attachments/assets/4cb7fee3-1d5f-46cf-8307-d4de2b177f67)
![1736697169854](https://github.com/user-attachments/assets/8eee6088-85f0-4af1-8529-69a03778b663)
![1736697169678](https://github.com/user-attachments/assets/361869ec-c0a9-4698-99cc-2c94a347d36f)

# Stemify: AI-Powered Multi-Instrument and Vocal Separation
## Overview
Stemify is an AI-driven application designed to separate mixed audio into individual stems—vocals, drums, bass, and other instruments—with minimal artifacts. Leveraging a phase-aware DeepMultiSourceUNet model trained on the MUSDB18 dataset, Stemify achieves high-quality separation suitable for music production, remixing, education, and DJing. A Flask-based web interface allows users to upload WAV/MP3 files, process them, and download isolated stems.

## Features
AI Stem Separation: Isolates vocals, drums, bass, and other instruments using a phase-aware U-Net model.
High Fidelity: Minimizes artifacts with a composite loss function (MSE, phase consistency, independence, magnitude sum).
Web Interface: Flask-based UI supports WAV/MP3 uploads, stem selection, and downloadable outputs.
Tempo & Key Detection: Planned module for BPM and key analysis to aid remixing.
Efficient Processing: Runs on modest hardware (1.6 GB VRAM GPU, 1.6 GiB RAM).

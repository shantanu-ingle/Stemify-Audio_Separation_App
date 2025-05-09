import os
import uuid
import shutil
import subprocess
import psutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import logging

# Import transcription pipelines
from bass_processing import process_bass_recording
from drum_processing import process_drum_recording
from vocal_processing import process_vocal_recording

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 44100
BASS_FREQ_RANGE = (40, 200)

def detect_key_tempo(file_path, sr=44100, tempo_range=(60, 180)):
    """
    Accurately detect the musical key and tempo of an audio file (WAV or MP3).
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate (default: 44100 Hz).
        tempo_range (tuple): Valid tempo range (min, max) in BPM.
    
    Returns:
        dict: {'key': str, 'tempo': float, 'key_confidence': float, 'tempo_confidence': float}
    """
    try:
        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.lower().endswith(('.wav', '.mp3')):
            raise ValueError("File must be .wav or .mp3")

        # Load audio file
        logging.info(f"Loading audio file for key/tempo detection: {file_path}")
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        logging.info(f"Audio loaded: duration={duration:.2f}s, sr={sr}")

        # Tempo detection
        # Use onset strength with smoothing for robust beat tracking
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        # Apply smoothing to reduce noise
        onset_env = librosa.util.normalize(onset_env)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512)
        # Refine tempo with multiple candidates
        tempos = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
        # Handle double/half tempo errors
        candidate_tempos = [tempo, tempo/2, tempo*2]
        valid_tempos = [t for t in candidate_tempos if tempo_range[0] <= t <= tempo_range[1]]
        if valid_tempos:
            tempo = np.median(valid_tempos)
        else:
            tempo = np.median(tempos)
        # Compute tempo confidence based on beat consistency
        beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr, hop_length=512))
        tempo_confidence = 1.0 / (np.std(beat_intervals) + 1e-8) if len(beat_intervals) > 0 else 0.0
        tempo_confidence = min(1.0, max(0.0, tempo_confidence / 10.0))  # Normalize to [0, 1]
        logging.info(f"Detected tempo: {tempo:.2f} BPM (confidence: {tempo_confidence:.2f})")

        # Key detection
        # Use harmonic component for clearer pitch analysis
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'))
        chroma_mean = np.mean(chroma, axis=1)  # Shape: (12,)
        # Krumhansl-Schmuckler key profiles
        key_profiles = np.zeros((24, 12))
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        for i in range(12):
            key_profiles[i] = np.roll(major_profile, i)
            key_profiles[i + 12] = np.roll(minor_profile, i)
        # Normalize profiles and chroma
        key_profiles = key_profiles / (np.std(key_profiles, axis=1, keepdims=True) + 1e-8)
        chroma_mean = chroma_mean / (np.std(chroma_mean) + 1e-8)
        # Compute correlations
        correlations = np.array([np.corrcoef(chroma_mean, profile)[0, 1] for profile in key_profiles])
        key_idx = np.argmax(correlations)
        key_names = [
            'C Major', 'C# Major', 'D Major', 'D# Major', 'E Major', 'F Major',
            'F# Major', 'G Major', 'G# Major', 'A Major', 'A# Major', 'B Major',
            'C Minor', 'C# Minor', 'D Minor', 'D# Minor', 'E Minor', 'F Minor',
            'F# Minor', 'G Minor', 'G# Minor', 'A Minor', 'A# Minor', 'B Minor'
        ]
        key = key_names[key_idx]
        # Compute key confidence
        key_confidence = correlations[key_idx] / (np.sum(correlations) + 1e-8)
        # Resolve minor/major ambiguity
        if 'Minor' in key:
            relative_major_idx = (key_idx - 12 - 3) % 12
            if correlations[relative_major_idx] > correlations[key_idx] * 0.95:
                key = key_names[relative_major_idx]
                key_confidence = correlations[relative_major_idx] / (np.sum(correlations) + 1e-8)
        logging.info(f"Detected key: {key} (confidence: {key_confidence:.2f})")

        return {
            'key': key,
            'tempo': round(tempo, 1),
            'key_confidence': round(key_confidence, 2),
            'tempo_confidence': round(tempo_confidence, 2)
        }

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {'key': 'Unknown', 'tempo': 0.0, 'key_confidence': 0.0, 'tempo_confidence': 0.0}

# Memory monitoring function
def print_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        ram = process.memory_info().rss / 1024**3
        logging.info(f"VRAM: {vram:.1f} MB, RAM: {ram:.2f} GB")
    except Exception as e:
        logging.error(f"Error in print_memory_usage: {e}")

# AttentionGate class
class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gating = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.theta_x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.phi_g = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, g):
        try:
            theta_x = self.theta_x(x)
            g_conv = self.gating(g)
            if g_conv.shape[2:] != theta_x.shape[2:]:
                g_conv = F.interpolate(g_conv, size=theta_x.shape[2:], mode='bilinear', align_corners=False)
            phi_g = self.phi_g(g_conv)
            f = F.relu(theta_x + phi_g, inplace=True)
            att = self.sigmoid(f)
            att_x = x * att
            att_x = self.conv(att_x)
            if att_x.shape[2:] != g.shape[2:]:
                att_x = F.interpolate(att_x, size=g.shape[2:], mode='bilinear', align_corners=False)
            return att_x
        except Exception as e:
            logging.error(f"Error in AttentionGate.forward: {e}")
            raise

# ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        try:
            residual = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x += residual
            return self.relu(x)
        except Exception as e:
            logging.error(f"Error in ResidualBlock.forward: {e}")
            raise

# DeepMultiSourceUNet class
class DeepMultiSourceUNet(nn.Module):
    def __init__(self):
        super(DeepMultiSourceUNet, self).__init__()
        self.encoder1 = self.conv_block(2, 64)
        self.res1 = ResidualBlock(64)
        self.encoder2 = self.conv_block(64, 128)
        self.res2 = ResidualBlock(128)
        self.encoder3 = self.conv_block(128, 256)
        self.res3 = ResidualBlock(256)
        self.encoder4 = self.conv_block(256, 512)
        self.res4 = ResidualBlock(512)
        self.encoder5 = self.conv_block(512, 768)
        self.res5 = ResidualBlock(768)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.att5 = AttentionGate(768, 768)
        self.decoder5 = self.upconv_block(1024, 768)
        self.res6 = ResidualBlock(768)
        self.att4 = AttentionGate(512, 512)
        self.decoder4 = self.upconv_block(768, 512)
        self.res7 = ResidualBlock(512)
        self.att3 = AttentionGate(256, 256)
        self.decoder3 = self.upconv_block(512, 256)
        self.res8 = ResidualBlock(256)
        self.att2 = AttentionGate(128, 128)
        self.decoder2 = self.upconv_block(256, 128)
        self.res9 = ResidualBlock(128)
        self.att1 = AttentionGate(64, 64)
        self.decoder1 = self.upconv_block(128, 64)
        self.res10 = ResidualBlock(64)

        self.final_conv = nn.Conv2d(64, 12, kernel_size=1, bias=True)
        nn.init.constant_(self.final_conv.bias, 0.1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        try:
            e1 = self.encoder1(x)
            e1 = self.res1(e1)
            e2 = self.encoder2(e1)
            e2 = self.res2(e2)
            e3 = self.encoder3(e2)
            e3 = self.res3(e3)
            e4 = self.encoder4(e3)
            e4 = self.res4(e4)
            e5 = self.encoder5(e4)
            e5 = self.res5(e5)
            b = self.bottleneck(e5)
            d5 = self.decoder5(b)
            e5_att = self.att5(e5, d5)
            d5 = d5 + e5_att
            d5 = self.res6(d5)
            d4 = self.decoder4(d5)
            e4_att = self.att4(e4, d4)
            d4 = d4 + e4_att
            d4 = self.res7(d4)
            d3 = self.decoder3(d4)
            e3_att = self.att3(e3, d3)
            d3 = d3 + e3_att
            d3 = self.res8(d3)
            d2 = self.decoder2(d3)
            e2_att = self.att2(e2, d2)
            d2 = d2 + e2_att
            d2 = self.res9(d2)
            d1 = self.decoder1(d2)
            e1_att = self.att1(e1, d1)
            d1 = d1 + e1_att
            d1 = self.res10(d1)
            output = self.final_conv(d1)
            if output.shape[2:] != x.shape[2:]:
                padding_h = x.shape[2] - output.shape[2]
                padding_w = x.shape[3] - output.shape[3]
                output = F.pad(output, (0, padding_w, 0, padding_h), mode='constant', value=0)
            magnitudes = torch.clamp(output[:, 0:4, :, :], 0.0, 5.0)
            sin_phases = torch.tanh(output[:, 4:8, :, :])
            cos_phases = torch.tanh(output[:, 8:12, :, :])
            norm = torch.sqrt(sin_phases**2 + cos_phases**2 + 1e-8)
            sin_phases = sin_phases / norm
            cos_phases = cos_phases / norm
            predicted_phases = torch.atan2(sin_phases, cos_phases)
            return magnitudes, predicted_phases
        except Exception as e:
            logging.error(f"Error in DeepMultiSourceUNet.forward: {e}")
            raise

def load_audio(path, sr=SAMPLE_RATE):
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
        logging.info(f"Loaded audio: {path}, shape={y.shape}")
        return y, sr
    except Exception as e:
        logging.error(f"Error loading audio {path}: {e}")
        raise

def compute_spectrogram(audio, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, chunk_size=1024):
    try:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
        mag = np.abs(stft)
        phase = np.angle(stft)
        log_mag = np.clip(np.log1p(mag), 0.0, 5.0)
        freq_bins, time_frames = mag.shape

        mag_chunks, phase_chunks, starts = [], [], []
        for i in range(0, time_frames, chunk_size):
            end = min(i + chunk_size, time_frames)
            m = log_mag[:, i:end]
            p = phase[:, i:end]
            if m.shape[1] < chunk_size:
                pad = chunk_size - m.shape[1]
                m = np.pad(m, ((0, 0), (0, pad)), mode='constant')
                p = np.pad(p, ((0, 0), (0, pad)), mode='constant')
            mag_chunks.append(m)
            phase_chunks.append(p)
            starts.append(i)

        logging.info(f"Computed spectrogram: chunks={len(mag_chunks)}")
        return mag_chunks, phase_chunks, stft, freq_bins, time_frames, starts
    except Exception as e:
        logging.error(f"Error computing spectrogram: {e}")
        raise

def reconstruct_audio(magnitude, phase, sr, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        mag = np.expm1(magnitude)
        stft = mag * np.exp(1j * phase)
        audio = librosa.istft(stft, hop_length=hop_length, window='hann', length=None)
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            audio /= max_amplitude
        else:
            logging.warning(f"Silent audio reconstructed")
        logging.info(f"Reconstructed audio: shape={audio.shape}, max_amplitude={max_amplitude:.4f}")
        return audio
    except Exception as e:
        logging.error(f"Error in reconstruct_audio: {e}")
        raise

def process_audio_file(model, audio_path, output_dir, device, n_fft=N_FFT, hop_length=HOP_LENGTH, chunk_size=1024, sr=SAMPLE_RATE):
    try:
        os.makedirs(output_dir, exist_ok=True)
        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_subdir = os.path.join(output_dir, song_name)
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"Processing {audio_path} into {output_subdir}")

        audio, sr = load_audio(audio_path, sr)
        mag_chunks, phase_chunks, mixture_stft, freq_bins, time_frames, chunk_starts = compute_spectrogram(audio, sr, n_fft, hop_length, chunk_size)
        mixture_mag = np.abs(mixture_stft)
        source_names = ['bass', 'vocals', 'drums', 'other']
        output_mags = [np.zeros((freq_bins, time_frames)) for _ in range(4)]
        output_phases = [np.zeros((freq_bins, time_frames)) for _ in range(4)]

        model.eval()
        with torch.no_grad():
            for mag_chunk, phase_chunk, start in tqdm(zip(mag_chunks, phase_chunks, chunk_starts), total=len(mag_chunks), desc="Processing chunks"):
                try:
                    mag_t = torch.tensor(mag_chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    ph_t = torch.tensor(phase_chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    inp = torch.cat([mag_t, ph_t], dim=1)
                    pred_mags, pred_phases = model(inp)
                    pm = pred_mags.squeeze(0).cpu().numpy()
                    pp = pred_phases.squeeze(0).cpu().numpy()
                    end = min(start + chunk_size, time_frames)
                    for i in range(4):
                        output_mags[i][:, start:end] = pm[i, :, :end - start]
                        output_phases[i][:, start:end] = pp[i, :, :end - start]
                    print_memory_usage()
                except Exception as e:
                    logging.error(f"Error processing chunk {start}: {e}")
                    raise

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        low_bin = np.argmax(freqs >= BASS_FREQ_RANGE[0])
        high_bin = np.argmax(freqs >= BASS_FREQ_RANGE[1]) if BASS_FREQ_RANGE[1] < max(freqs) else freq_bins

        masks = np.zeros((4, freq_bins, time_frames))
        for j in range(4):
            m = output_mags[j]
            mask = np.clip(m / (mixture_mag + 1e-8), 0, 1)
            mask = uniform_filter1d(mask, size=5, axis=1)
            masks[j] = mask

        mask_sum = masks.sum(axis=0) + 1e-8
        norm_masks = masks / mask_sum
        mask_drums = norm_masks[2]

        power_bass = output_mags[0]**2
        power_others = sum(o**2 for i, o in enumerate(output_mags) if i != 0)
        fw = np.zeros(freq_bins)
        fw[low_bin:high_bin] = 1
        contrast = (power_bass * fw[:, None]) / (power_bass * fw[:, None] + power_others + 1e-8)
        mask_bass = np.clip(uniform_filter1d(contrast**1.5, size=5, axis=1), 0, 1)

        output_files = {}
        for idx, name in enumerate(source_names):
            try:
                if name == 'drums':
                    stft_masked = mixture_stft * mask_drums
                    audio_out = librosa.istft(stft_masked, hop_length=hop_length, window='hann', length=None)
                elif name == 'bass':
                    stft_masked = mixture_stft * mask_bass
                    audio_out = librosa.istft(stft_masked, hop_length=hop_length, window='hann', length=None)
                else:
                    audio_out = reconstruct_audio(output_mags[idx], output_phases[idx], sr, n_fft, hop_length)

                max_amplitude = np.max(np.abs(audio_out))
                if max_amplitude > 0:
                    audio_out /= max_amplitude
                else:
                    logging.warning(f"Silent audio for {name}")

                path = os.path.join(output_subdir, f"{name}.wav")
                sf.write(path, audio_out, sr, 'PCM_24')
                if os.path.exists(path):
                    output_files[name] = path
                    logging.info(f"Saved {name} to {path}")
                else:
                    logging.error(f"Failed to save {name} WAV file at {path}")
            except Exception as e:
                logging.error(f"Error processing source {name}: {e}")
                continue

        # Save processing ID and song name for later use
        with open(os.path.join(output_subdir, "metadata.txt"), "w") as f:
            f.write(f"proc_id={os.path.basename(output_dir)}\nsong_name={song_name}")

        return output_files
    except Exception as e:
        logging.error(f"Error in process_audio_file: {e}")
        raise

# Flask App + Routes
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\Users\OMEN\Saved Programs\Projects\Frontend new\try-app - Copy\backend\multisource_unet_phase_aware_v36.pth"
try:
    model = DeepMultiSourceUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info(f"Loaded model from {model_path}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            return jsonify({'error': 'Invalid file type'}), 400

        proc_id = str(uuid.uuid4())
        base_dir = os.path.join(os.path.dirname(__file__), 'static', 'outputs', proc_id)
        os.makedirs(base_dir, exist_ok=True)

        infile = os.path.join(base_dir, file.filename)
        file.save(infile)
        logging.info(f"Saved input file: {infile}")

        # Detect key and tempo of the original input song
        key_tempo_result = detect_key_tempo(infile)
        metadata = {
            'key': key_tempo_result['key'],
            'tempo': key_tempo_result['tempo']
        }
        logging.info(f"Key and tempo detection result: {metadata}")

        stems = process_audio_file(model, infile, base_dir, device)
        host = request.host_url.rstrip('/')
        song_name = os.path.splitext(file.filename)[0]
        files = {
            name: f"{host}/static/outputs/{proc_id}/{song_name}/{name}.wav"
            for name in ['bass', 'vocals', 'drums', 'other']
        }
        notes = {
            name: f"{host}/view-notes/{proc_id}/{song_name}/{name}"
            for name in ['bass', 'drums', 'vocals']
        }

        logging.info(f"Returning files: {files}, notes: {notes}, metadata: {metadata}")
        return jsonify({
            'files': files,
            'notes': notes,
            'proc_id': proc_id,
            'song_name': song_name,
            'metadata': metadata
        }), 200
    except Exception as e:
        logging.error(f"Error in process_audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view-notes/<proc_id>/<song_name>/<source>', methods=['GET'])
def view_notes(proc_id, song_name, source):
    try:
        if source not in ['bass', 'drums', 'vocals']:
            return jsonify({'error': 'Invalid source'}), 400

        # Construct WAV file path
        wav_path = os.path.join(os.path.dirname(__file__), 'static', 'outputs', proc_id, song_name, f"{source}.wav")
        if not os.path.exists(wav_path):
            return jsonify({'error': f"{source}.wav not found"}), 404

        # Construct PDF output path
        pdf_name = f"output_{source}.pdf"
        pdf_path = os.path.join(r"C:\Users\OMEN\Saved Programs\Projects\Frontend new\Combined app\output", pdf_name)

        # Check if PDF already exists
        if os.path.exists(pdf_path):
            logging.info(f"Serving existing PDF: {pdf_path}")
            return send_from_directory(
                r"C:\Users\OMEN\Saved Programs\Projects\Frontend new\Combined app\output",
                pdf_name,
                as_attachment=False,
                mimetype='application/pdf'
            )

        # Trigger transcription
        transcription_func = {
            'bass': process_bass_recording,
            'drums': process_drum_recording,
            'vocals': process_vocal_recording
        }[source]
        result = transcription_func(wav_path)
        if result and os.path.exists(result):
            logging.info(f"Generated and serving PDF: {result}")
            return send_from_directory(
                r"C:\Users\OMEN\Saved Programs\Projects\Frontend new\Combined app\output",
                pdf_name,
                as_attachment=False,
                mimetype='application/pdf'
            )
        else:
            logging.error(f"Failed to generate PDF for {source}")
            return jsonify({'error': f"Failed to generate PDF for {source}"}), 500
    except Exception as e:
        logging.error(f"Error in view_notes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/outputs/<path:filename>')
def serve_static(filename):
    try:
        root = os.path.join(os.path.dirname(__file__), 'static', 'outputs')
        return send_from_directory(root, filename)
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}")
        return jsonify({'error': f'File not found: {filename}'}), 404

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'outputs'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
import numpy as np
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'float'):
    np.float = np.float64

import librosa
import music21
from scipy.signal import medfilt, butter, filtfilt
import os
import subprocess

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Base output directory
OUTPUT_DIR = r"C:\Users\OMEN\Saved Programs\Projects\Frontend new\Combined app\output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to MuseScore executable
MUSESCORE_EXE = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# Verify MuseScore exists
if not os.path.exists(MUSESCORE_EXE):
    logging.warning(f"MuseScore not found at {MUSESCORE_EXE}. PDF generation may fail.")

def butter_lowpass(cutoff, fs, order=5):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    except Exception as e:
        logging.error(f"Error in butter_lowpass: {e}")
        raise

def butter_lowpass_filter(data, cutoff, fs, order=5):
    try:
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        logging.error(f"Error in butter_lowpass_filter: {e}")
        raise

def get_tempo_advanced(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=44100)
        y_filtered = librosa.effects.preemphasis(y, coef=0.95)
        y_harmonic, y_percussive = librosa.effects.hpss(y_filtered)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env)
        return float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0] if len(tempo) > 0 else 120)
    except Exception as e:
        logging.error(f"Error in get_tempo_advanced for {audio_path}: {e}")
        return 120

def closest_bass_note(freq):
    try:
        bass_note_frequencies = {
            "E1": 41.20, "F1": 43.65, "F#1": 46.25, "G1": 49.00, "G#1": 51.91,
            "A1": 55.00, "A#1": 58.27, "B1": 61.74,
            "C2": 65.41, "C#2": 69.30, "D2": 73.42, "D#2": 77.78, "E2": 82.41,
            "F2": 87.31, "F#2": 92.50, "G2": 98.00, "G#2": 103.83,
            "A2": 110.00, "A#2": 116.54, "B2": 123.47,
            "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81,
            "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65,
            "A3": 220.00, "A#3": 233.08, "B3": 246.94,
            "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
            "F4": 349.23, "F#4": 369.99, "G4": 392.00
        }
        return min(bass_note_frequencies, key=lambda note: abs(bass_note_frequencies[note] - freq))
    except Exception as e:
        logging.error(f"Error in closest_bass_note: {e}")
        return "E2"

def round_duration(duration, quantum=0.25):
    try:
        return round(duration / quantum) * quantum
    except Exception as e:
        logging.error(f"Error in round_duration: {e}")
        return duration

def merge_notes(quantized_notes):
    try:
        if not quantized_notes:
            return []
        merged = [quantized_notes[0]]
        for current in quantized_notes[1:]:
            prev_pitch, prev_duration = merged[-1]
            curr_pitch, curr_duration = current
            if curr_pitch == prev_pitch:
                merged[-1] = (prev_pitch, prev_duration + curr_duration)
            else:
                merged.append(current)
        return merged
    except Exception as e:
        logging.error(f"Error in merge_notes: {e}")
        return quantized_notes

def process_bass_audio(file_path):
    try:
        logging.info(f"Processing bass audio: {file_path}")
        y, sr = librosa.load(file_path, sr=44100, mono=True)
        max_amplitude = np.max(np.abs(y))
        if max_amplitude == 0:
            logging.error(f"Silent audio file: {file_path}")
            return None
        y = y / max_amplitude

        y_bass_focused = butter_lowpass_filter(y, cutoff=400, fs=sr, order=5)
        bpm = get_tempo_advanced(file_path)
        logging.info(f"Estimated Tempo: {bpm:.2f} BPM")
        beat_duration = 60.0 / bpm

        onset_frames = librosa.onset.onset_detect(
            y=y_bass_focused, sr=sr, backtrack=True,
            pre_max=10, post_max=10, pre_avg=100, post_avg=100, delta=0.03, wait=12
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_bass_focused, fmin=30, fmax=400, sr=sr, frame_length=2048, hop_length=512
            )
        except AttributeError:
            logging.warning("pYin not available, falling back to yin")
            f0 = librosa.yin(
                y_bass_focused, fmin=30, fmax=400, sr=sr, frame_length=2048, hop_length=512
            )
            voiced_flag = np.ones_like(f0, dtype=bool)

        f0 = np.nan_to_num(f0)
        f0 = medfilt(f0, kernel_size=7)
        pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

        quantized_notes = []
        if len(onset_times) <= 1:
            logging.warning("Few onsets detected, creating artificial segments")
            segment_duration = 0.25
            num_segments = int(librosa.get_duration(y=y, sr=sr) / segment_duration)
            onset_times = np.linspace(0, librosa.get_duration(y=y, sr=sr), num_segments + 1)

        for i in range(len(onset_times) - 1):
            start, end = onset_times[i], onset_times[i + 1]
            indices = np.where((pitch_times >= start) & (pitch_times < end))[0]
            if len(indices) == 0:
                continue
            confident_indices = indices[voiced_flag[indices]] if len(voiced_flag) == len(f0) else indices
            if len(confident_indices) > 0:
                indices = confident_indices
            valid_f0 = f0[indices]
            valid_f0 = valid_f0[valid_f0 > 0]
            if len(valid_f0) == 0:
                continue
            hist, bin_edges = np.histogram(valid_f0, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            most_common_freq = bin_centers[np.argmax(hist)]
            most_common_note = closest_bass_note(most_common_freq)
            octave = int(most_common_note[-1])
            if octave > 3:
                most_common_freq /= 2
                most_common_note = closest_bass_note(most_common_freq)
            duration = (end - start) / beat_duration
            qlength = round_duration(duration, quantum=0.125)
            if qlength < 0.125:
                continue
            quantized_notes.append((most_common_note, qlength))

        if len(onset_times) > 0:
            start = onset_times[-1]
            end = librosa.get_duration(y=y, sr=sr)
            indices = np.where((pitch_times >= start) & (pitch_times <= end))[0]
            if len(indices) > 0:
                confident_indices = indices[voiced_flag[indices]] if len(voiced_flag) == len(f0) else indices
                if len(confident_indices) > 0:
                    indices = confident_indices
                valid_f0 = f0[indices]
                valid_f0 = valid_f0[valid_f0 > 0]
                if len(valid_f0) > 0:
                    hist, bin_edges = np.histogram(valid_f0, bins=20)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    most_common_freq = bin_centers[np.argmax(hist)]
                    most_common_note = closest_bass_note(most_common_freq)
                    octave = int(most_common_note[-1])
                    if octave > 3:
                        most_common_freq /= 2
                        most_common_note = closest_bass_note(most_common_freq)
                    duration = (end - start) / beat_duration
                    qlength = round_duration(duration, quantum=0.125)
                    if qlength >= 0.125:
                        quantized_notes.append((most_common_note, qlength))

        logging.info(f"Detected {len(quantized_notes)} raw notes before merging")
        if quantized_notes:
            logging.info("Sample of detected notes:")
            for note in quantized_notes[:5]:
                logging.info(f"{note[0]}, {note[1]} beats")

        merged_notes = merge_notes(quantized_notes)
        logging.info(f"After merging: {len(merged_notes)} notes")

        score = music21.stream.Score()
        metadata = music21.metadata.Metadata()
        metadata.title = "Bass Transcription"
        score.metadata = metadata
        bass_part = music21.stream.Part()
        bass_instrument = music21.instrument.ElectricBass()
        bass_part.insert(0, bass_instrument)
        bass_clef = music21.clef.BassClef()
        bass_part.insert(0, bass_clef)
        tempo_mark = music21.tempo.MetronomeMark(number=bpm)
        bass_part.insert(0, tempo_mark)
        time_signature = music21.meter.TimeSignature('4/4')
        bass_part.insert(0, time_signature)

        if merged_notes:
            current_measure_duration = 0
            measure = music21.stream.Measure(number=1)
            for note_idx, (note_name, qlength) in enumerate(merged_notes):
                n = music21.note.Note(note_name)
                n.quarterLength = qlength
                if current_measure_duration + qlength > 4.0:
                    remaining = 4.0 - current_measure_duration
                    if remaining > 0:
                        first_part = music21.note.Note(note_name)
                        first_part.quarterLength = remaining
                        first_part.tie = music21.tie.Tie('start')
                        measure.append(first_part)
                        bass_part.append(measure)
                        measure = music21.stream.Measure(number=measure.number + 1)
                        second_part = music21.note.Note(note_name)
                        second_part.quarterLength = qlength - remaining
                        second_part.tie = music21.tie.Tie('stop')
                        measure.append(second_part)
                        current_measure_duration = qlength - remaining
                    else:
                        bass_part.append(measure)
                        measure = music21.stream.Measure(number=measure.number + 1)
                        measure.append(n)
                        current_measure_duration = qlength
                else:
                    measure.append(n)
                    current_measure_duration += qlength
                    if current_measure_duration == 4.0:
                        bass_part.append(measure)
                        measure = music21.stream.Measure(number=measure.number + 1)
                        current_measure_duration = 0
                if note_idx == len(merged_notes) - 1 and current_measure_duration > 0:
                    bass_part.append(measure)

        score.append(bass_part)
        output_xml_path = os.path.join(OUTPUT_DIR, "output_bass.xml")

        for n in score.recurse().notes:
            if n.duration.quarterLength <= 0:
                logging.warning(f"Fixing invalid duration for {n.nameWithOctave}")
                n.duration.quarterLength = 0.25

        try:
            score = score.makeNotation()
            score.write('musicxml', fp=output_xml_path)
            logging.info(f"MusicXML file saved: {output_xml_path}")
            return output_xml_path
        except Exception as e:
            logging.error(f"Error writing MusicXML: {e}")
            return None
    except Exception as e:
        logging.error(f"Error in process_bass_audio: {e}")
        return None

def generate_pdf(musicxml_path):
    try:
        output_pdf_path = os.path.join(OUTPUT_DIR, "output_bass.pdf")
        cmd = [MUSESCORE_EXE, musicxml_path, "-o", output_pdf_path]
        logging.info(f"Running MuseScore command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logging.info(f"MuseScore stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"MuseScore stderr: {result.stderr}")
        if os.path.exists(output_pdf_path):
            logging.info(f"PDF bass score saved: {output_pdf_path}")
            return output_pdf_path
        else:
            logging.error("PDF generation failed")
            return None
    except Exception as e:
        logging.error(f"Error executing MuseScore: {e}")
        return None

def process_bass_recording(audio_path):
    try:
        musicxml_file = process_bass_audio(audio_path)
        if musicxml_file:
            pdf_file = generate_pdf(musicxml_file)
            return pdf_file
        return None
    except Exception as e:
        logging.error(f"Error in process_bass_recording: {e}")
        return None
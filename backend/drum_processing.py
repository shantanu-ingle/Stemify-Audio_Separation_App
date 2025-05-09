import numpy as np
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'float'):
    np.float = np.float64

import librosa
import music21
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
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

def get_tempo_advanced(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=44100)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(
            y=y_percussive, sr=sr, hop_length=512, aggregate=np.median, fmax=8000
        )
        onset_env = gaussian_filter1d(onset_env, sigma=2)
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=512, start_bpm=120, tightness=100
        )
        return float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0] if len(tempo) > 0 else 120)
    except Exception as e:
        logging.error(f"Error in get_tempo_advanced for {audio_path}: {e}")
        return 120

def classify_drum_sound(y, sr, onset_time, duration=0.1):
    try:
        start_sample = int(onset_time * sr)
        end_sample = min(len(y), start_sample + int(duration * sr))
        if start_sample >= len(y) or start_sample >= end_sample:
            return "unknown"
        drum_segment = y[start_sample:end_sample]
        if len(drum_segment) == 0:
            return "unknown"
        spectral_centroid = librosa.feature.spectral_centroid(y=drum_segment, sr=sr)[0].mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=drum_segment, sr=sr)[0].mean()
        rms = librosa.feature.rms(y=drum_segment)[0].mean()
        zcr = librosa.feature.zero_crossing_rate(y=drum_segment)[0].mean()
        if spectral_centroid < 1000 and rms > 0.1:
            return "kick"
        elif 1000 <= spectral_centroid < 3000 and zcr > 0.1:
            return "snare"
        elif spectral_centroid > 5000 and zcr > 0.2:
            return "hi-hat"
        elif 800 <= spectral_centroid < 3000 and 0.05 < rms < 0.15:
            return "tom"
        elif spectral_centroid > 4000 and spectral_rolloff > 8000:
            return "crash" if rms > 0.1 else "ride"
        return "unknown"
    except Exception as e:
        logging.error(f"Error in classify_drum_sound: {e}")
        return "unknown"

def round_duration(beat_time, quantum=0.25):
    try:
        return round(beat_time / quantum) * quantum
    except Exception as e:
        logging.error(f"Error in round_duration: {e}")
        return beat_time

def quantize_onset_times(onset_times, bpm, quantum=0.25):
    try:
        beat_duration = 60.0 / bpm
        beat_times = [time / beat_duration for time in onset_times]
        quantized_beat_times = [round_duration(beat, quantum) for beat in beat_times]
        return [time * beat_duration for time in quantized_beat_times]
    except Exception as e:
        logging.error(f"Error in quantize_onset_times: {e}")
        return onset_times

DRUM_TO_MIDI = {
    "kick": 36, "snare": 38, "hi-hat": 42, "tom": 45, "crash": 49, "ride": 51, "unknown": 37
}

def process_drum_audio(file_path):
    try:
        logging.info(f"Processing drum audio: {file_path}")
        y, sr = librosa.load(file_path, sr=44100, mono=True)
        max_amplitude = np.max(np.abs(y))
        if max_amplitude == 0:
            logging.error(f"Silent audio file: {file_path}")
            return None
        y = y / max_amplitude

        bpm = get_tempo_advanced(file_path)
        logging.info(f"Estimated Tempo: {bpm:.2f} BPM")
        beat_duration = 60.0 / bpm

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, fmax=8000, hop_length=512)
        onset_env = medfilt(onset_env, kernel_size=3)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512, backtrack=True,
            pre_max=5, post_max=5, pre_avg=10, post_avg=10, delta=0.07, wait=8
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        quantized_times = quantize_onset_times(onset_times, bpm, quantum=0.125)
        drum_hits = []
        for i, onset_time in enumerate(onset_times):
            drum_type = classify_drum_sound(y, sr, onset_time)
            quantized_time = quantized_times[i]
            beat_position = quantized_time / beat_duration
            drum_hits.append((drum_type, beat_position))

        grouped_hits = {}
        for drum_type, beat_pos in drum_hits:
            if beat_pos not in grouped_hits:
                grouped_hits[beat_pos] = []
            grouped_hits[beat_pos].append(drum_type)
        sorted_positions = sorted(grouped_hits.keys())

        logging.info(f"Detected {len(drum_hits)} drum hits, {len(sorted_positions)} unique positions")
        if drum_hits:
            logging.info("Sample of detected drum hits:")
            for i, pos in enumerate(sorted_positions[:5]):
                logging.info(f"Beat {pos:.2f}: {', '.join(grouped_hits[pos])}")

        score = music21.stream.Score()
        drum_part = music21.stream.Part()
        perc_clef = music21.clef.PercussionClef()
        drum_part.append(perc_clef)
        tempo_mark = music21.tempo.MetronomeMark(number=bpm)
        drum_part.append(tempo_mark)
        time_signature = music21.meter.TimeSignature('4/4')
        drum_part.append(time_signature)

        if sorted_positions:
            measure = music21.stream.Measure()
            measure.append(perc_clef)
            measure.append(time_signature)
            measure.append(tempo_mark)
            current_measure_position = 0.0
            measure_count = 1
            last_position = 0.0

            for beat_pos in sorted_positions:
                measure_pos = beat_pos % 4.0
                if int(beat_pos / 4.0) > int(last_position / 4.0):
                    measure.number = measure_count
                    drum_part.append(measure)
                    measure_count += 1
                    measure = music21.stream.Measure()
                    current_measure_position = beat_pos % 4.0
                else:
                    current_measure_position = measure_pos

                drum_types = grouped_hits[beat_pos]
                if len(drum_types) > 1:
                    notes = []
                    for drum_type in drum_types:
                        midi_num = DRUM_TO_MIDI.get(drum_type, DRUM_TO_MIDI["unknown"])
                        n = music21.note.Note(midi_num)
                        n.notehead = 'x' if drum_type in ['hi-hat', 'crash', 'ride'] else 'normal'
                        notes.append(n)
                    chord = music21.chord.Chord(notes)
                    chord.quarterLength = 0.25
                    measure.insert(current_measure_position, chord)
                else:
                    drum_type = drum_types[0]
                    midi_num = DRUM_TO_MIDI.get(drum_type, DRUM_TO_MIDI["unknown"])
                    n = music21.note.Note(midi_num)
                    n.quarterLength = 0.25
                    n.notehead = 'x' if drum_type in ['hi-hat', 'crash', 'ride'] else 'normal'
                    measure.insert(current_measure_position, n)

                last_position = beat_pos

            if measure.notes:
                measure.number = measure_count
                drum_part.append(measure)

            score.append(drum_part)
            output_xml_path = os.path.join(OUTPUT_DIR, "output_drums.xml")

            for n in score.recurse().getElementsByClass(music21.note.Note):
                if not n.duration.type or n.duration.type == 'zero':
                    logging.warning(f"Assigning default type 'quarter' to note {n.pitch}")
                    n.duration.type = 'quarter'
            for c in score.recurse().getElementsByClass(music21.chord.Chord):
                if not c.duration.type or c.duration.type == 'zero':
                    pitches = [p.nameWithOctave for p in c.pitches]
                    logging.warning(f"Assigning default type 'quarter' to chord {pitches}")
                    c.duration.type = 'quarter'

            try:
                score = score.makeNotation(inPlace=False)
            except Exception as e:
                logging.error(f"Error during makeNotation: {e}")

            score.write("musicxml", fp=output_xml_path)
            logging.info(f"MusicXML file saved: {output_xml_path}")
            return output_xml_path
        else:
            logging.warning("No valid drum hits detected")
            return None
    except Exception as e:
        logging.error(f"Error in process_drum_audio: {e}")
        return None

def generate_pdf(musicxml_path):
    try:
        output_pdf_path = os.path.join(OUTPUT_DIR, "output_drums.pdf")
        cmd = [MUSESCORE_EXE, musicxml_path, "-o", output_pdf_path]
        logging.info(f"Running MuseScore command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logging.info(f"MuseScore stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"MuseScore stderr: {result.stderr}")
        if os.path.exists(output_pdf_path):
            logging.info(f"PDF drum score saved: {output_pdf_path}")
            return output_pdf_path
        else:
            logging.error("PDF generation failed")
            return None
    except Exception as e:
        logging.error(f"Error executing MuseScore: {e}")
        return None

def analyze_drum_pattern(drum_hits, bpm):
    try:
        positions = [pos for _, pos in drum_hits]
        if len(positions) < 2:
            return {"pattern_type": "unknown", "complexity": "n/a"}
        iois = np.diff(positions)
        median_ioi = np.median(iois)
        beat_duration = 60.0 / bpm
        beats_per_second = bpm / 60.0
        notes_per_second = len(positions) / (positions[-1] - positions[0]) if positions[-1] > positions[0] else 0
        density_ratio = notes_per_second / beats_per_second
        complexity = "simple" if density_ratio < 1.2 else "moderate" if density_ratio < 2.0 else "complex"
        kick_positions = [pos for type, pos in drum_hits if type == "kick"]
        pattern_type = "unknown"
        if kick_positions:
            kicks_on_1_and_3 = sum(1 for pos in kick_positions if pos % 1.0 < 0.1 or (pos % 2.0) < 0.1)
            pattern_type = "standard rock" if kicks_on_1_and_3 > len(kick_positions) * 0.7 else "varied"
        return {
            "pattern_type": pattern_type,
            "complexity": complexity,
            "notes_per_second": notes_per_second,
            "median_ioi": median_ioi
        }
    except Exception as e:
        logging.error(f"Error in analyze_drum_pattern: {e}")
        return {"pattern_type": "unknown", "complexity": "n/a"}

def process_drum_recording(audio_path):
    try:
        musicxml_file = process_drum_audio(audio_path)
        if musicxml_file:
            pdf_file = generate_pdf(musicxml_file)
            return pdf_file
        return None
    except Exception as e:
        logging.error(f"Error in process_drum_recording: {e}")
        return None
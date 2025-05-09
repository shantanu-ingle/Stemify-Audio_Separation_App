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
            y=y_harmonic, sr=sr, hop_length=512, aggregate=np.mean, fmax=4000
        )
        onset_env = gaussian_filter1d(onset_env, sigma=3)
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=512, start_bpm=100, tightness=80
        )
        return float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0] if len(tempo) > 0 else 100)
    except Exception as e:
        logging.error(f"Error in get_tempo_advanced for {audio_path}: {e}")
        return 100

def extract_pitch_contour(y, sr):
    try:
        hop_length = 512
        frame_length = 2048
        fmin = 75.0
        fmax = 800.0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length, frame_length=frame_length, fill_na=np.nan
        )
        pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
        return pitch_times, f0
    except Exception as e:
        logging.error(f"Error in extract_pitch_contour: {e}")
        raise

def convert_hz_to_note(freq_hz):
    if np.isnan(freq_hz) or freq_hz <= 0:
        return None, None, None
    try:
        midi_note = 69 + 12 * np.log2(freq_hz / 440.0)
        closest_midi = round(midi_note)
        cents_deviation = 100 * (midi_note - closest_midi)
        note_name = music21.pitch.Pitch(closest_midi).nameWithOctave
        return closest_midi, note_name, cents_deviation
    except Exception as e:
        logging.error(f"Error in convert_hz_to_note: {e}")
        return None, None, None

def segment_into_notes(pitch_times, pitch_values, onset_times, min_note_duration=0.1):
    try:
        notes = []
        if len(onset_times) < 2:
            logging.warning("Insufficient onset times for segmentation")
            return notes
        onset_times = np.append(onset_times, pitch_times[-1] + 0.1)
        for i in range(len(onset_times) - 1):
            start_time = onset_times[i]
            end_time = onset_times[i + 1]
            if end_time - start_time < min_note_duration:
                continue
            mask = (pitch_times >= start_time) & (pitch_times < end_time)
            segment_pitch_values = pitch_values[mask]
            valid_pitches = segment_pitch_values[~np.isnan(segment_pitch_values)]
            if len(valid_pitches) == 0:
                continue
            median_pitch = np.nanmedian(segment_pitch_values)
            midi_note, note_name, cents_deviation = convert_hz_to_note(median_pitch)
            if midi_note is not None:
                duration = end_time - start_time
                notes.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'midi_note': midi_note,
                    'note_name': note_name,
                    'freq_hz': median_pitch,
                    'cents_deviation': cents_deviation
                })
        return notes
    except Exception as e:
        logging.error(f"Error in segment_into_notes: {e}")
        raise

def process_vocal_audio(file_path, lyrics=None):
    try:
        logging.info(f"Processing vocal audio: {file_path}")
        y, sr = librosa.load(file_path, sr=44100, mono=True)
        max_amplitude = np.max(np.abs(y))
        if max_amplitude == 0:
            logging.error(f"Silent audio file: {file_path}")
            return None
        y = y / max_amplitude

        bpm = get_tempo_advanced(file_path)
        logging.info(f"Estimated Tempo: {bpm:.2f} BPM")
        beat_duration = 60.0 / bpm

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, fmax=6000, hop_length=512)
        onset_env = medfilt(onset_env, kernel_size=5)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512, backtrack=True,
            pre_max=7, post_max=7, pre_avg=15, post_avg=15, delta=0.05, wait=10
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        pitch_times, pitch_values = extract_pitch_contour(y, sr)
        notes = segment_into_notes(pitch_times, pitch_values, onset_times)

        logging.info(f"Detected {len(notes)} notes")
        if notes:
            logging.info("Sample of detected notes:")
            for i, note in enumerate(notes[:5]):
                logging.info(f"Note {i+1}: {note['note_name']}, start={note['start_time']:.2f}s, duration={note['duration']:.2f}s")

        beat_duration = 60.0 / bpm
        for note in notes:
            beat_start = note['start_time'] / beat_duration
            beat_end = note['end_time'] / beat_duration
            quantized_start = round(beat_start * 4) / 4
            quantized_end = round(beat_end * 4) / 4
            if quantized_end <= quantized_start:
                quantized_end = quantized_start + 0.25
            note['quantized_start_time'] = quantized_start * beat_duration
            note['quantized_end_time'] = quantized_end * beat_duration
            note['quantized_duration'] = note['quantized_end_time'] - note['quantized_start_time']
            note['quantized_beat_start'] = quantized_start
            note['quantized_beat_end'] = quantized_end
            note['quantized_beat_duration'] = quantized_end - quantized_start

        score = music21.stream.Score()
        vocal_part = music21.stream.Part()
        treble_clef = music21.clef.TrebleClef()
        vocal_part.append(treble_clef)
        tempo_mark = music21.tempo.MetronomeMark(number=bpm)
        vocal_part.append(tempo_mark)
        time_signature = music21.meter.TimeSignature('4/4')
        vocal_part.append(time_signature)

        if notes:
            measure = music21.stream.Measure()
            measure.append(treble_clef)
            measure.append(time_signature)
            measure.append(tempo_mark)
            current_measure_position = 0.0
            measure_count = 1
            last_position = 0.0
            sorted_notes = sorted(notes, key=lambda x: x['quantized_beat_start'])

            for i, note_data in enumerate(sorted_notes):
                beat_pos = note_data['quantized_beat_start']
                note_duration = note_data['quantized_beat_duration']
                midi_note = note_data['midi_note']
                if midi_note is None:
                    continue
                measure_pos = beat_pos % 4.0
                if int(beat_pos / 4.0) > int(last_position / 4.0):
                    measure.number = measure_count
                    vocal_part.append(measure)
                    measure_count += 1
                    measure = music21.stream.Measure()
                    current_measure_position = beat_pos % 4.0
                else:
                    current_measure_position = measure_pos

                n = music21.note.Note(midi_note)
                n.quarterLength = note_duration
                if lyrics and i < len(lyrics):
                    n.lyric = lyrics[i]
                measure.insert(current_measure_position, n)
                last_position = beat_pos

            if measure.notes:
                measure.number = measure_count
                vocal_part.append(measure)

            score.append(vocal_part)
            output_xml_path = os.path.join(OUTPUT_DIR, "output_vocals.xml")

            for n in score.recurse().getElementsByClass(music21.note.Note):
                if not n.duration.type or n.duration.type == 'zero':
                    logging.warning(f"Assigning default type 'quarter' to note {n.pitch}")
                    n.duration.type = 'quarter'

            try:
                score = score.makeNotation(inPlace=False)
            except Exception as e:
                logging.error(f"Error during makeNotation: {e}")

            score.write("musicxml", fp=output_xml_path)
            logging.info(f"MusicXML file saved: {output_xml_path}")
            return output_xml_path
        else:
            logging.warning("No valid notes detected")
            return None
    except Exception as e:
        logging.error(f"Error in process_vocal_audio: {e}")
        return None

def generate_pdf(musicxml_path):
    try:
        output_pdf_path = os.path.join(OUTPUT_DIR, "output_vocals.pdf")
        cmd = [MUSESCORE_EXE, musicxml_path, "-o", output_pdf_path]
        logging.info(f"Running MuseScore command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logging.info(f"MuseScore stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"MuseScore stderr: {result.stderr}")
        if os.path.exists(output_pdf_path):
            logging.info(f"PDF vocal score saved: {output_pdf_path}")
            return output_pdf_path
        else:
            logging.error("PDF generation failed")
            return None
    except Exception as e:
        logging.error(f"Error executing MuseScore: {e}")
        return None

def analyze_vocal_performance(notes, bpm):
    try:
        if len(notes) < 2:
            return {"quality": "insufficient data", "pitch_stability": "n/a", "range": "n/a"}
        midi_notes = [note['midi_note'] for note in notes if note['midi_note'] is not None]
        durations = [note['duration'] for note in notes]
        deviations = [abs(note['cents_deviation']) for note in notes if note['cents_deviation'] is not None]
        if not midi_notes:
            return {"quality": "no valid pitches detected", "pitch_stability": "n/a", "range": "n/a"}
        min_note = min(midi_notes)
        max_note = max(midi_notes)
        range_semitones = max_note - min_note
        avg_deviation = sum(deviations) / len(deviations) if deviations else float('nan')
        range_classification = "unknown"
        avg_pitch = sum(midi_notes) / len(midi_notes)
        if avg_pitch < 50:
            range_classification = "bass"
        elif avg_pitch < 60:
            range_classification = "baritone"
        elif avg_pitch < 70:
            range_classification = "tenor/alto"
        else:
            range_classification = "soprano"
        stability_rating = "unknown"
        if avg_deviation < 15:
            stability_rating = "excellent"
        elif avg_deviation < 30:
            stability_rating = "good"
        elif avg_deviation < 50:
            stability_rating = "moderate"
        else:
            stability_rating = "poor"
        quality = "moderate"
        if range_semitones > 24 and stability_rating in ["excellent", "good"]:
            quality = "excellent"
        elif range_semitones > 12 and stability_rating in ["excellent", "good", "moderate"]:
            quality = "good"
        elif stability_rating == "poor" or range_semitones < 8:
            quality = "needs improvement"
        return {
            "quality": quality,
            "pitch_stability": stability_rating,
            "avg_cents_deviation": avg_deviation,
            "range_semitones": range_semitones,
            "voice_type": range_classification,
            "lowest_note": music21.pitch.Pitch(min_note).nameWithOctave,
            "highest_note": music21.pitch.Pitch(max_note).nameWithOctave
        }
    except Exception as e:
        logging.error(f"Error in analyze_vocal_performance: {e}")
        return {"quality": "error", "pitch_stability": "n/a", "range": "n/a"}

def align_lyrics_to_notes(notes, lyrics_text):
    try:
        if not lyrics_text or not notes:
            return []
        words = lyrics_text.strip().split()
        syllables = []
        for word in words:
            vowels = 'aeiouy'
            v_count = 0
            in_vowel_group = False
            for char in word.lower():
                if char in vowels:
                    if not in_vowel_group:
                        v_count += 1
                        in_vowel_group = True
                else:
                    in_vowel_group = False
            syllable_count = max(1, v_count)
            if syllable_count == 1:
                syllables.append(word)
            else:
                chars_per_syllable = max(1, len(word) // syllable_count)
                for i in range(0, len(word), chars_per_syllable):
                    end = min(i + chars_per_syllable, len(word))
                    syllables.append(word[i:end] + ("-" if end < len(word) else ""))
        aligned_lyrics = [syllables[i] if i < len(syllables) else "" for i in range(len(notes))]
        return aligned_lyrics
    except Exception as e:
        logging.error(f"Error in align_lyrics_to_notes: {e}")
        return []

def process_vocal_recording(audio_path, lyrics=None):
    try:
        musicxml_file = process_vocal_audio(audio_path, lyrics)
        if musicxml_file:
            pdf_file = generate_pdf(musicxml_file)
            return pdf_file
        return None
    except Exception as e:
        logging.error(f"Error in process_vocal_recording: {e}")
        return None
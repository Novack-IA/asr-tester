import os
import time
import json
import random
import pandas as pd
import jiwer
import assemblyai as aai
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
config = aai.TranscriptionConfig(
                language_code="pt",
                speech_model=aai.SpeechModel.best
                )

ROOT_DIR = "test"
CSV_PATH = "metadata_test_final.csv"
LOG_FILE = "transcription_log_assembly.txt"
SAMPLE_SIZE = 100  

reference_data = pd.read_csv(CSV_PATH)

reference_dict = {os.path.basename(path): text for path, text in zip(reference_data["file_path"], reference_data["text"])}

def get_audio_duration(file_path):
    """Obtém a duração real do áudio em segundos."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000 
    except Exception as e:
        print(f"Erro ao obter duração de {file_path}: {str(e)}")
        return None

def transcribe_file(file_path):
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(file_path, config=config)
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Erro ao transcrever {file_path}: {transcript.error}")
            return None, None
        return transcript.text, transcript.audio_duration
    except Exception as e:
        print(f"Erro ao transcrever {file_path}: {str(e)}")
        return None, None

def evaluate_transcriptions(reference, hypothesis):
    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    return {"WER": wer, "CER": cer}

def log_transcription(file_name, reference_text, transcribed_text, metrics, processing_time, audio_duration):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Arquivo: {file_name}\n")
        log.write(f"Referência: {reference_text}\n")
        log.write(f"Transcrição: {transcribed_text}\n")
        log.write(f"Duração do áudio: {audio_duration:.2f} segundos\n")
        log.write(f"WER: {metrics['WER']:.4f}, CER: {metrics['CER']:.4f}\n")
        log.write(f"Tempo de processamento: {processing_time:.2f} segundos\n")
        log.write("-" * 50 + "\n")

def process_dataset():
    all_files = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                all_files.append(os.path.join(root, file))
    
    sampled_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    results = []
    valid_metrics = []
    
    for file_path in sampled_files:
        file_name = os.path.basename(file_path)
        if file_name not in reference_dict:
            print(f"Sem transcrição de referência para {file_name}, ignorando...")
            continue
        
        try:
            start_time = time.time()
            config = aai.TranscriptionConfig(
                language_code="pt",
                speech_model=aai.SpeechModel.best
                )

            transcription, audio_duration = transcribe_file(file_path)
            end_time = time.time()
            
            if transcription is None:
                continue
            
            reference_text = reference_dict[file_name]
            metrics = evaluate_transcriptions(reference_text, transcription)
            processing_time = end_time - start_time
            
            if metrics["WER"] < 1.0 and metrics["CER"] < 1.0:  # Ignorar erros 100%
                valid_metrics.append((metrics["WER"], metrics["CER"], processing_time, audio_duration))
            
            results.append({
                "file_path": file_path,
                "file_name": file_name,
                "reference_text": reference_text,
                "transcribed_text": transcription,
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "WER": metrics["WER"],
                "CER": metrics["CER"]
            })
            
            log_transcription(file_name, reference_text, transcription, metrics, processing_time, audio_duration)
            
            print(f"Transcrito: {file_path}")
            print(f"WER: {metrics['WER']:.2f}, CER: {metrics['CER']:.2f}")
            print(f"Tempo de processamento: {processing_time:.2f} segundos")
        except Exception as e:
            print(f"Erro ao transcrever {file_path}: {str(e)}")
    
    return results, valid_metrics

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("Log de Transcrição - AssemblyAI\n" + "=" * 50 + "\n")
    
    results, valid_metrics = process_dataset()
    
    if valid_metrics:
        avg_wer = sum(w for w, _, _, _ in valid_metrics) / len(valid_metrics)
        avg_cer = sum(c for _, c, _, _ in valid_metrics) / len(valid_metrics)
        avg_processing_time = sum(t for _, _, t, _ in valid_metrics) / sum(a for _, _, _, a in valid_metrics)
    else:
        avg_wer, avg_cer, avg_processing_time = 0, 0, 0
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Média WER: {avg_wer:.4f}\n")
        log.write(f"Média CER: {avg_cer:.4f}\n")
        log.write(f"Tempo médio de processamento por segundo de áudio: {avg_processing_time:.4f} s/s\n")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("results_assembly.csv", index=False, encoding="utf-8")
    
    print("Avaliação concluída! Resultados salvos em results_assembly.csv e log de transcrição em transcription_log_assembly.txt")

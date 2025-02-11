import os
import time
import random
import argparse
import pandas as pd
import jiwer
from dotenv import load_dotenv
from pydub import AudioSegment
from mark_preprocessing import MarkPreprocessing
import torch

# Importação condicional dos modelos
try:
    import whisper
except ImportError:
    whisper = None

try:
    from deepgram import DeepgramClient, PrerecordedOptions
except ImportError:
    DeepgramClient = None

try:
    import assemblyai as aai
except ImportError:
    aai = None

load_dotenv()

ROOT_DIR = "test"
CSV_PATH = "metadata_test_final.csv"
LOG_FILE = "transcription_log.txt"
SAMPLE_SIZE = 10

reference_data = pd.read_csv(CSV_PATH)
reference_dict = {os.path.basename(path): text for path, text in zip(reference_data["file_path"], reference_data["text"])}

normalizer = MarkPreprocessing()

def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000
    except Exception as e:
        print(f"Erro ao obter duração de {file_path}: {str(e)}")
        return None

def transcribe_deepgram(file_path):
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
    with open(file_path, "rb") as audio:
        options = PrerecordedOptions(model="nova-2", language="pt-BR", punctuate=True, filler_words= True)
        start_time = time.time()
        response = deepgram.listen.prerecorded.v("1").transcribe_file({"buffer": audio, "mimetype": "audio/wav"}, options)
        end_time = time.time()
    return response["results"]["channels"][0]["alternatives"][0]["transcript"], end_time - start_time

def transcribe_assemblyai(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, config=aai.TranscriptionConfig(language_code="pt", speech_model=aai.SpeechModel.best))
    if transcript.status == aai.TranscriptStatus.error:
        return None, None
    return transcript.text, transcript.audio_duration

def transcribe_whisper(file_path):
    model = whisper.load_model("small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    start_time = time.time()
    result = model.transcribe(file_path, language="pt")
    end_time = time.time()
    return result["text"].strip(), end_time - start_time


def evaluate_transcriptions(reference, hypothesis):
    normalized_reference = normalizer(reference) or ""
    normalized_hypothesis = normalizer(hypothesis) or ""
    
    if not normalized_hypothesis:
        return {"WER": 100.0, "CER": 100.0, "normalized_reference": normalized_reference, "normalized_hypothesis": normalized_hypothesis}
    
    wer = jiwer.wer(normalized_reference, normalized_hypothesis) * 100
    cer = jiwer.cer(normalized_reference, normalized_hypothesis) * 100
    return {"WER": wer, "CER": cer, "normalized_reference": normalized_reference, "normalized_hypothesis": normalized_hypothesis}

def log_transcription(file_name, reference_text, transcribed_text, metrics, audio_duration, processing_time, model_name):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Arquivo: {file_name}\n")
        log.write(f"Referência sem norm: {reference_text}\n")
        log.write(f"Transcrição sem norm: {transcribed_text}\n")
        log.write(f"Referência: {metrics["normalized_reference"]}\n")
        log.write(f"Transcrição: {metrics["normalized_hypothesis"]}\n")
        log.write(f"WER: {metrics['WER']:.2f}%, CER: {metrics['CER']:.2f}%\n")
        log.write(f"Duração do áudio: {audio_duration:.2f} segundos\n")
        log.write(f"Tempo de processamento: {processing_time:.2f} segundos\n")
        log.write(f"Modelo Utilizado: {model_name}\n")
        log.write("-" * 50 + "\n")

def process_dataset(model_name):
    transcribe_func = {
        "deepgramnova2": transcribe_deepgram,
        "assemblyaibest": transcribe_assemblyai,
        "whispersmall": transcribe_whisper
    }.get(model_name)
    
    if not transcribe_func:
        print("Modelo não reconhecido. Escolha entre 'deepgramnova2', 'assemblyaibest' ou 'whispersmall'.")
        return
    
    all_files = [os.path.join(root, file) for root, _, files in os.walk(ROOT_DIR) for file in files if file.endswith((".wav", ".mp3"))]
    sampled_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    results, valid_metrics = [], []
    
    for file_path in sampled_files:
        file_name = os.path.basename(file_path)
        if file_name not in reference_dict:
            continue
        
        try:
            transcription, duration = transcribe_func(file_path)
            reference_text = reference_dict[file_name]
            metrics = evaluate_transcriptions(reference_text, transcription)
            audio_duration = get_audio_duration(file_path)
            if audio_duration is None:
                continue
            
            valid_metrics.append((metrics["WER"], metrics["CER"], duration, audio_duration))
            results.append({
                "file_path": file_path,
                "file_name": file_name,
                "reference_text": reference_text,
                "transcribed_text": transcription,
                "audio_duration": audio_duration,
                "processing_time": duration,
                "WER": metrics["WER"],
                "CER": metrics["CER"],
                "model": model_name
            })
            
            log_transcription(file_name, reference_text, transcription, metrics, audio_duration, duration, model_name)
        except Exception as e:
            print(f"Erro ao transcrever {file_path}: {str(e)}")
    
    return results, valid_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de transcrição com diferentes modelos.")
    parser.add_argument("--model", type=str, required=True, help="Escolha entre 'deepgramnova2', 'assemblyaibest' ou 'whispersmall'")
    args = parser.parse_args()
    
    results, valid_metrics = process_dataset(args.model)
    
    if valid_metrics:
        total_inference_time = sum(t for _, _, t, _ in valid_metrics)
        total_audio_time = sum(a for _, _, _, a in valid_metrics)
        avg_wer = sum(w for w, _, _, _ in valid_metrics) / len(valid_metrics)
        avg_cer = sum(c for _, c, _, _ in valid_metrics) / len(valid_metrics)
        avg_inference_time = total_inference_time / total_audio_time if total_audio_time > 0 else 0
        total_minutes_processed = total_audio_time / 60
    else:
        avg_wer, avg_cer, avg_inference_time, total_minutes_processed = 0, 0, 0, 0
    
    valid_metrics_filtered = [(w, c) for w, c, _, _ in valid_metrics if not (w == 100.0 and c == 100.0)]
    
    if valid_metrics_filtered:
        avg_wer_filtered = sum(w for w, _ in valid_metrics_filtered) / len(valid_metrics_filtered)
        avg_cer_filtered = sum(c for _, c in valid_metrics_filtered) / len(valid_metrics_filtered)
    else:
        avg_wer_filtered, avg_cer_filtered = 0, 0
    
    summary_data = {
        "Model": [args.model],
        "WER (Médio)": [avg_wer],
        "CER (Médio)": [avg_cer],
        "WER (Médio, sem 100%)": [avg_wer_filtered],
        "CER (Médio, sem 100%)": [avg_cer_filtered],
        "Tempo Total de Inferência (s)": [total_inference_time],
        "Tempo Médio de Inferência": [avg_inference_time],
        "Minutos Totais Processados": [total_minutes_processed]
    }
    summary_df = pd.DataFrame(summary_data)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"results-todelete/results_{args.model}.csv", index=False, encoding="utf-8")
    
    summary_file = "summary.csv"
    if os.path.exists(summary_file):
        summary_df.to_csv(summary_file, mode='a', header=False, index=False, encoding="utf-8")
    else:
        summary_df.to_csv(summary_file, index=False, encoding="utf-8")
    
    print(f"Avaliação concluída! Resultados salvos em results_{args.model}.csv e resumo atualizado em summary.csv")

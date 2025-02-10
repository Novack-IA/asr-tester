import os
import time
import json
import pandas as pd
import jiwer
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

ROOT_DIR = "test"
CSV_PATH = "metadata_test_final.csv"
LOG_FILE = "transcription_log.txt"

reference_data = pd.read_csv(CSV_PATH)

reference_dict = {os.path.basename(path): text for path, text in zip(reference_data["file_path"], reference_data["text"])}

def transcribe_file(file_path):
    with open(file_path, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        options = PrerecordedOptions(model="nova-2", language="pt-BR")
        
        start_time = time.time()
        response = deepgram.listen.prerecorded.v("1").transcribe_file(source, options)
        end_time = time.time()
        
        transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        duration = end_time - start_time
        
        return transcription, duration

def evaluate_transcriptions(reference, hypothesis):
    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    return {"WER": wer, "CER": cer}

def log_transcription(file_name, reference_text, transcribed_text, metrics, processing_time):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Arquivo: {file_name}\n")
        log.write(f"Referência: {reference_text}\n")
        log.write(f"Transcrição: {transcribed_text}\n")
        log.write(f"WER: {metrics['WER']:.4f}, CER: {metrics['CER']:.4f}\n")
        log.write(f"Tempo de processamento: {processing_time:.2f} segundos\n")
        log.write("-" * 50 + "\n")

def process_dataset():
    results = []
    
    for root, dirs, files in os.walk(ROOT_DIR):  
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file) 
                file_name = os.path.basename(file_path) 
                
                if file_name not in reference_dict:
                    print(f"Sem transcrição de referência para {file_name}, ignorando...")
                    continue
                
                try:
                    transcription, duration = transcribe_file(file_path)
                    reference_text = reference_dict[file_name]
                    metrics = evaluate_transcriptions(reference_text, transcription)
                    
                    results.append({
                        "file_path": file_path,
                        "file_name": file_name,
                        "reference_text": reference_text,
                        "transcribed_text": transcription,
                        "processing_time": duration,
                        "WER": metrics["WER"],
                        "CER": metrics["CER"]
                    })
                    
                    log_transcription(file_name, reference_text, transcription, metrics, duration)
                    
                    print(f"Transcrito: {file_path}")
                    print(f"WER: {metrics['WER']:.2f}, CER: {metrics['CER']:.2f}")
                    print(f"Tempo de processamento: {duration:.2f} segundos")
                except Exception as e:
                    print(f"Erro ao transcrever {file_path}: {str(e)}")
    
    return results

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("Log de Transcrição\n" + "=" * 50 + "\n")
    
    results = process_dataset()
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("results.csv", index=False, encoding="utf-8")
    
    print("Avaliação concluída! Resultados salvos em results.json, results.csv e log de transcrição em transcription_log.txt")

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
import httpx

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

ROOT_DIR = "test"  # Pasta que contém as subpastas (alip, CORAL, NURC_RE, sp, Ted_part1, Ted_part3)
CSV_PATH = "metadata_test_final.csv"
LOG_FILE = "transcription_log.txt"

# Carrega os dados de referência e cria um dicionário mapeando o nome do arquivo para o texto
reference_data = pd.read_csv(CSV_PATH)
reference_dict = {os.path.basename(path): text for path, text in zip(reference_data["file_path"], reference_data["text"])}

normalizer = MarkPreprocessing()

def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # duração em segundos
    except Exception as e:
        print(f"Erro ao obter duração de {file_path}: {str(e)}")
        return None

def transcribe_deepgram(file_path):
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
    with open(file_path, "rb") as audio:
        options = PrerecordedOptions(model="nova-2", language="pt-BR", punctuate=True, filler_words=True)
        start_time = time.time()
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio, "mimetype": "audio/wav"},
            options,
            timeout=httpx.Timeout(900.0, connect=10.0)
        )
        end_time = time.time()
    return response["results"]["channels"][0]["alternatives"][0]["transcript"], end_time - start_time

def transcribe_assemblyai(file_path):
    start_time = time.time()
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        file_path,
        config=aai.TranscriptionConfig(language_code="pt", speech_model=aai.SpeechModel.best)
    )
    if transcript.status == aai.TranscriptStatus.error:
        return None, None
    end_time = time.time()
    return transcript.text, end_time - start_time

def transcribe_whisper(file_path):
    model = whisper.load_model("small")
    start_time = time.time()
    result = model.transcribe(file_path, language="pt")
    end_time = time.time()
    return result["text"].strip(), end_time - start_time

def evaluate_transcriptions(reference, hypothesis):
    normalized_reference = normalizer(reference) or ""
    normalized_hypothesis = normalizer(hypothesis) or ""
    
    if not normalized_hypothesis:
        return {
            "WER": 100.0,
            "CER": 100.0,
            "normalized_reference": normalized_reference,
            "normalized_hypothesis": normalized_hypothesis
        }
    
    wer = jiwer.wer(normalized_reference, normalized_hypothesis) * 100
    cer = jiwer.cer(normalized_reference, normalized_hypothesis) * 100
    return {
        "WER": wer,
        "CER": cer,
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis
    }

def log_transcription(file_name, reference_text, transcribed_text, metrics, audio_duration, processing_time, model_name, dataset_token):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Arquivo: {file_name}\n")
        log.write(f"Dataset: {dataset_token}\n")
        log.write(f"Referência sem norm: {reference_text}\n")
        log.write(f"Transcrição sem norm: {transcribed_text}\n")
        log.write(f"Referência: {metrics['normalized_reference']}\n")
        log.write(f"Transcrição: {metrics['normalized_hypothesis']}\n")
        log.write(f"WER: {metrics['WER']:.2f}%, CER: {metrics['CER']:.2f}%\n")
        log.write(f"Duração do áudio: {audio_duration:.2f} segundos\n")
        log.write(f"Tempo de processamento: {processing_time:.2f} segundos\n")
        log.write(f"Modelo Utilizado: {model_name}\n")
        log.write("-" * 50 + "\n")

# Função para selecionar os arquivos de áudio conforme a configuração escolhida
def get_sampled_files(config, sample_size, root_dir, subdataset=None):
    if config == "atual":
        # Seleciona aleatoriamente n arquivos dentre todos os arquivos dentro de 'test' (incluindo subpastas)
        all_files = [os.path.join(root, file)
                     for root, _, files in os.walk(root_dir)
                     for file in files if file.endswith((".wav", ".mp3"))]
        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
    elif config == "proporcional":
        # Para cada subpasta dentro de 'test', seleciona exatamente n arquivos (ou o máximo disponível)
        sampled_files = []
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                files_in_sub = [os.path.join(subfolder_path, file)
                                for file in os.listdir(subfolder_path)
                                if file.endswith((".wav", ".mp3"))]
                if files_in_sub:
                    sampled_files.extend(random.sample(files_in_sub, min(sample_size, len(files_in_sub))))
    elif config == "unitaria":
        # Seleciona n arquivos de uma única subpasta específica
        if subdataset is None:
            raise ValueError("Para a configuração unitária, é necessário especificar o subdataset.")
        dataset_path = os.path.join(root_dir, subdataset)
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Subdataset {subdataset} não encontrado em {root_dir}.")
        files_in_sub = [os.path.join(dataset_path, file)
                        for file in os.listdir(dataset_path)
                        if file.endswith((".wav", ".mp3"))]
        sampled_files = random.sample(files_in_sub, min(sample_size, len(files_in_sub)))
    else:
        raise ValueError("Configuração desconhecida.")
    return sampled_files

# Agora, a função process_dataset recebe os parâmetros de configuração do dataset
def process_dataset(model_name, config, sample_size, subdataset=None):
    transcribe_func = {
        "deepgramnova2": transcribe_deepgram,
        "assemblyaibest": transcribe_assemblyai,
        "whispersmall": transcribe_whisper
    }.get(model_name)
    
    if not transcribe_func:
        print("Modelo não reconhecido. Escolha entre 'deepgramnova2', 'assemblyaibest' ou 'whispersmall'.")
        return
    
    # Seleciona os arquivos conforme a configuração escolhida
    sampled_files = get_sampled_files(config, sample_size, ROOT_DIR, subdataset)
    results, valid_metrics = [], []
    
    for file_path in sampled_files:
        file_name = os.path.basename(file_path)
        # Obtém o token do dataset a partir do nome da subpasta
        dataset_token = os.path.basename(os.path.dirname(file_path))
        # Se não houver referência para o arquivo, pula-o
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
                "dataset": dataset_token,
                "reference_text": reference_text,
                "transcribed_text": transcription,
                "audio_duration": audio_duration,
                "processing_time": duration,
                "WER": metrics["WER"],
                "CER": metrics["CER"],
                "model": model_name
            })
            
            log_transcription(file_name, reference_text, transcription, metrics, audio_duration, duration, model_name, dataset_token)
        except Exception as e:
            print(f"Erro ao transcrever {file_path}: {str(e)}")
    
    return results, valid_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teste de transcrição com diferentes modelos e configurações de dataset."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Escolha entre 'deepgramnova2', 'assemblyaibest' ou 'whispersmall'")
    parser.add_argument("--config", type=str, default="atual",
                        choices=["atual", "proporcional", "unitaria"],
                        help=("Configuração do dataset: 'atual' (amostra global aleatória), "
                              "'proporcional' (n áudios de cada subdataset) ou "
                              "'unitaria' (n áudios de um subdataset específico)"))
    parser.add_argument("--sample", type=int, default=10,
                        help="Número de áudios a serem amostrados (por configuração)")
    parser.add_argument("--subdataset", type=str, default=None,
                        help="Nome do subdataset a ser utilizado (somente para configuração 'unitaria')")
    args = parser.parse_args()
    
    # Validação para configuração unitária
    if args.config == "unitaria" and not args.subdataset:
        parser.error("Para a configuração 'unitaria', é necessário especificar o argumento --subdataset.")
    
    results, valid_metrics = process_dataset(args.model, args.config, args.sample, args.subdataset)
    
    if valid_metrics:
        total_inference_time = round(sum(t for _, _, t, _ in valid_metrics), 2)
        total_audio_time = round(sum(a for _, _, _, a in valid_metrics), 2)
        avg_wer = round(sum(w for w, _, _, _ in valid_metrics) / len(valid_metrics), 2)
        avg_cer = round(sum(c for _, c, _, _ in valid_metrics) / len(valid_metrics), 2)
        avg_inference_time = round(total_inference_time / total_audio_time, 2) if total_audio_time > 0 else 0
        total_minutes_processed = round(total_audio_time / 60, 2)
    else:
        avg_wer, avg_cer, avg_inference_time, total_minutes_processed = 0, 0, 0, 0
    
    valid_metrics_filtered = [(w, c) for w, c, _, _ in valid_metrics if not (w == 100.0 and c == 100.0)]
    
    if valid_metrics_filtered:
        avg_wer_filtered = round(sum(w for w, _ in valid_metrics_filtered) / len(valid_metrics_filtered), 2)
        avg_cer_filtered = round(sum(c for _, c in valid_metrics_filtered) / len(valid_metrics_filtered), 2)
    else:
        avg_wer_filtered, avg_cer_filtered = 0, 0
    
    # Prepara os dados de resumo incluindo tokens para a configuração e o subdataset (se aplicável)
    summary_data = {
        "Model": [args.model],
        "Dataset Config": [args.config],
        "Subdataset": [args.subdataset if args.config == "unitaria" else "todos"],
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
    
    # Atualiza (ou cria) o arquivo de resultados "results.csv" sem apagar os resultados anteriores
    results_file = "results.csv"
    if os.path.exists(results_file):
        df_results.to_csv(results_file, mode='a', header=False, index=False, encoding="utf-8")
    else:
        df_results.to_csv(results_file, index=False, encoding="utf-8")
    
    # Atualiza (ou cria) o arquivo de resumo "summary.csv" sem apagar os resumos anteriores
    summary_file = "summary.csv"
    if os.path.exists(summary_file):
        summary_df.to_csv(summary_file, mode='a', header=False, index=False, encoding="utf-8")
    else:
        summary_df.to_csv(summary_file, index=False, encoding="utf-8")
    
    print(f"Avaliação concluída! Resultados salvos em {results_file} e resumo atualizado em {summary_file}")

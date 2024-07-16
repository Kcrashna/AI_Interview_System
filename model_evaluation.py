import torch
import sacrebleu
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd

def evaluate_accuracy(model, tokenizer, dataset):
    model.eval()
    references = []
    predictions = []

    all_targets = []
    all_predictions = []

    for index, row in dataset.iterrows():
        input_text = row['input_text']
        target_text = row['target_text']
        inputs = tokenizer.encode_plus(input_text, return_tensors='pt', max_length=512, truncation=True, padding=True)

        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=4, early_stopping=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        references.append(target_text)
        predictions.append(generated_text)

        all_targets.append(target_text)
        all_predictions.append(generated_text)

    # BLEU Score
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    # Rouge Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    # Perplexity
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for index, row in dataset.iterrows():
        input_text = row['input_text']
        target_text = row['target_text']
        inputs = tokenizer.encode_plus(input_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        labels = tokenizer.encode_plus(target_text, return_tensors='pt', max_length=512, truncation=True, padding=True)['input_ids']

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    perplexity = torch.exp(torch.tensor(total_loss / len(dataset)))

    # Accuracy, Precision, Recall, F1 Score
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    # Confusion Matrix
    confusion_mat = confusion_matrix(all_targets, all_predictions)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_targets, all_predictions)
    roc_auc = auc(fpr, tpr)

    return {
        'bleu': bleu.score,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL,
        'perplexity': perplexity.item(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_mat,
        'roc_curve': (fpr, tpr),
        'auc': roc_auc
    }

def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

import torch
import sacrebleu
from transformers import T5ForConditionalGeneration, T5Tokenizer

def evaluate_accuracy(model, tokenizer, dataset):
    model.eval()
    references = []
    predictions = []

    for index, row in dataset.iterrows():
        input_text = row['input_text']
        target_text = row['target_text']
        inputs = tokenizer.encode_plus(input_text, return_tensors='pt', max_length=512, truncation=True, padding=True)

        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=4, early_stopping=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        references.append(target_text)
        predictions.append(generated_text)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score

def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split

def tokenize_function(example, tokenizer):
    inputs = tokenizer(example['input_text'], padding='max_length', truncation=True, max_length=512)
    targets = tokenizer(example['target_text'], padding='max_length', truncation=True, max_length=64)
    return {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': targets.input_ids
    }

def train_model(df, model_name='t5-small', output_dir='./results', epochs=10, batch_size=8):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    train_dataset = train_df.apply(lambda x: tokenize_function(x, tokenizer), axis=1).tolist()
    val_dataset = val_df.apply(lambda x: tokenize_function(x, tokenizer), axis=1).tolist()

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

import data_loader
import model_training
import model_evaluation
import interactive_qa

# Load and preprocess the data
df = data_loader.load_data('/content/drive/MyDrive/pp/merged.json')

# Train the model
model_training.train_model(df, model_name='t5-small', output_dir='./content/drive/MyDrive/pp/question_generator_model')

# Evaluate the model
model, tokenizer = model_evaluation.load_model('./content/drive/MyDrive/pp/question_generator_model')
accuracy = model_evaluation.evaluate_accuracy(model, tokenizer, df)
print(f"BLEU Score on Validation Set: {accuracy:.2f}")

# Run interactive Q&A
interactive_qa.interactive_qa(df, model, tokenizer)

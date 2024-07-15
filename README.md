# T5 Question Generation and Interactive interview System

## Description
This project trains a T5 model for question generation based on provided answers and difficulty levels. It also includes an interactive Q&A system that adapts the difficulty of questions based on the user's performance.

## Files
- `data_loader.py`: Contains the data loading and preprocessing functions.
- `model_training.py`: Contains the model training functions.
- `model_evaluation.py`: Contains the model evaluation functions.
- `interactive_qa.py`: Contains the interactive Q&A system functions.
- `main.py`: The main script to run the entire pipeline.

## Usage
1. Ensure you have the necessary dependencies installed.
2. Adjust file paths as necessary.
3. Run the main script:
    ```bash
    python main.py
    ```

## Dependencies
- pandas
- transformers
- torch
- nltk
- sacrebleu

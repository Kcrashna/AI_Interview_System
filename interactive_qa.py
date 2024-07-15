
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from nltk.tokenize import word_tokenize
from question_filter import filter_questions  # Import the function

def generate_question(answer, difficulty):
    input_text = f"generate question: {answer} </s> difficulty: {difficulty}"
    inputs = tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=64, num_return_sequences=1)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def evaluate_answer(user_answer, correct_answer, keywords):
    user_tokens = word_tokenize(user_answer.lower())
    correct_tokens = word_tokenize(correct_answer.lower())
    common_tokens = set(user_tokens).intersection(set(correct_tokens))
    keyword_tokens = set(user_tokens).intersection(set(keywords))
    if correct_tokens and keywords:
        return (len(common_tokens) / len(correct_tokens) + len(keyword_tokens) / len(keywords)) * 50
    return 0

def interactive_qa(df, model, tokenizer):
    difficulty_levels = ['Easy', 'Medium', 'Hard']
    difficulty_index = 0
    score = 0
    total_questions = 0
    correct_answers_streak = 0

    job_description = input("Enter job description (backend/frontend/mern stack): ").strip().lower()

    try:
        filtered_df = filter_questions(df, job_description)  # Use the filter_questions function
    except ValueError as e:
        print(e)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    while difficulty_index < len(difficulty_levels):
        difficulty = difficulty_levels[difficulty_index]
        questions = filtered_df[(filtered_df['difficulty'].str.lower() == difficulty.lower())].to_dict('records')

        for q in questions:
            total_questions += 1
            print(f"\nQuestion {total_questions} (Difficulty: {difficulty}):")
            print(q['question'])

            user_answer = input("Your answer: ")
            if user_answer.lower() == 'quit':
                print(f"\nFinal score: {score:.2f}/{total_questions * 100}")
                return

            answer_score = evaluate_answer(user_answer, q['answer'], q['keywords'])
            score += answer_score
            print(f"Score for this answer: {answer_score:.2f}%")
            print(f"Correct answer: {q['answer']}")

            if answer_score > 50:
                correct_answers_streak += 1
            else:
                correct_answers_streak = 0

            if difficulty == 'Easy' and correct_answers_streak >= 3:
                difficulty_index = 1  # Switch to Medium
                correct_answers_streak = 0
                break
            elif difficulty == 'Medium' and correct_answers_streak >= 3:
                difficulty_index = 2  # Switch to Hard
                correct_answers_streak = 0
                break
            elif difficulty == 'Medium' and correct_answers_streak == 0:
                difficulty_index = 0  # Switch back to Easy
                break
            elif difficulty == 'Hard' and correct_answers_streak == 0:
                difficulty_index = 1  # Switch back to Medium
                break

            next_question = generate_question(user_answer, difficulty)
            print(f"Next question: {next_question}")

    print(f"\nFinal score: {score:.2f}/{total_questions * 100}")

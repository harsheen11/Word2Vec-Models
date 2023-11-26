import csv
import gensim.downloader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open('synonym.txt', 'r') as file:
    lines = file.readlines()

questions = []
correct_answers = []
options = []

for i in range(0, len(lines), 6):
    question = lines[i].strip().split('.')[1].strip()
    correct_answer = lines[i + 5].strip()
    choices = [lines[j].strip().split('.')[1].strip() for j in range(i + 1, i + 5)]
    questions.append(question)
    correct_answers.append(correct_answer)
    options.append(choices)

model_name = ['word2vec-google-news-300','glove-wiki-gigaword-200','glove-twitter-200','glove-wiki-gigaword-100','glove-wiki-gigaword-300']

with open('analysis.csv', 'w', newline='') as analysis_file:
    pass

for model_index in range(len(model_name)):
    model = gensim.downloader.load(model_name[model_index])
    results = []
    correct_count = 0
    without_guess_count = 0

    for i in range(len(questions)):
        question = questions[i]
        correct_answer = correct_answers[i]
        choices = options[i]

        if all(word in model.key_to_index for word in [question] + choices):
            question_vec = np.array([model[question]])
            choice_vecs = np.array([model[word] for word in choices])
            similarities = cosine_similarity(question_vec, choice_vecs)[0]
            predicted_index = np.argmax(similarities)

            if 0 <= predicted_index < len(choices):
                predicted_answer = choices[predicted_index]
            else:
                predicted_answer = 'guess'

            if predicted_answer == choices[ord(correct_answer) - ord('a')]:
                label = 'correct'
                correct_count += 1
            else:
                label = 'wrong'

            without_guess_count += 1

        else:
            label = 'guess'
            predicted_answer = 'guess'

        results.append((question, choices[ord(correct_answer) - ord('a')], predicted_answer, label))

    with open(f'{model_name[model_index]}-details.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)

    accuracy = correct_count / without_guess_count if without_guess_count > 0 else 0

    with open('analysis.csv', 'a', newline='') as analysis_file:
        writer = csv.writer(analysis_file)
        writer.writerow([model_name[model_index], len(model.key_to_index), correct_count, without_guess_count, accuracy])

    print(f"{model_name[model_index]}-details.csv made")

print("Program complete")

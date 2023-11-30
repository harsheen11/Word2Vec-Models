import csv
import gensim.downloader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

analysis_data = pd.read_csv('analysis.csv', names=['Model', 'Len Key To Index', 'Correct', 'No Guess', 'Accuracy'])
models = analysis_data['Model']
correct_val = analysis_data['Correct']
no_guess_val = analysis_data['No Guess']
accuracy_val = analysis_data['Accuracy']
fig, ax1 = plt.subplots(figsize=(16, 8))
bar_width = 0.25
index = range(len(models))

bar1 = ax1.bar(index, correct_val, bar_width, label='Correct', color='skyblue')
ax1.set_xlabel('Model')
ax1.set_ylabel('Correct', color='black')
ax1.set_title('Analysis of Models')
for i, value in enumerate(correct_val):
    ax1.text(i, value + 0.1, str(value), ha='center', va='bottom')

ax2 = ax1.twinx()
ax2.set_ylabel('No Guess', color='black')
bar2 = ax2.bar([i + bar_width for i in index], no_guess_val, bar_width, label='No Guess', color='lightcoral')
for i, value in enumerate(no_guess_val):
    ax2.text(i + bar_width, value + 0.1, str(value), ha='center', va='bottom')

ax3 = ax1.twinx()
ax3.set_ylabel('Accuracy', color='black')
ax3.spines['right'].set_position(('outward', 60))
bar3 = ax3.bar([i + 2 * bar_width for i in index], accuracy_val, bar_width, label='Accuracy', color='lightgreen')
for i, value in enumerate(accuracy_val):
    ax3.text(i + 2 * bar_width, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
ax2.legend(loc='upper right', bbox_to_anchor=(0.2, 1.1))
ax3.legend(loc='upper right', bbox_to_anchor=(0.3, 1.1))

plt.savefig('analysis_of_models.png', dpi=300)

print("Program complete")

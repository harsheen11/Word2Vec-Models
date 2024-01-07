# COMP472_Assignment2
Python program that uses different pre-trained models and our own made model to answer the Synonym Test automatically. Simply run the code to genrate outputs assuming the datasets are present in the project. 


To train our model we used 16 textbooks with enrich vocabulary for better results in prediction. 
In the below image, the first four models were pre-trained models and the rest were our own models trained from same corpus but with different embedding sizes to analyze the results effectively. 
For results we can observe high accuracy for pre-trained models from Wiki and Twitter ranging from 85%-90% but for our own trained models on a comparitively smaller corpus, accuracy was quite low. Many words were labelled as "No guess" meaning that there could be found no synonym for such words in the corpus. 
 
![analysis_of_models](https://github.com/harsheen11/Word2Vec-Models/assets/87078923/50bcc20b-3076-411a-8d3d-4986aab62da9)

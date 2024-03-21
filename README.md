# Naive-Bayes-Email-Classification
This project introduces an email classifier developed to categorize emails into two distinct classes: spam (unwanted emails) and not-spam (legitimate emails, also known as ham). 

# Email Classification with Naive Bayes
# Overview
This project introduces an email classifier developed to categorize emails into two distinct classes: spam (unwanted emails) and not-spam (legitimate emails, also known as ham). Leveraging the principles of Naive Bayes classification, the program learns to differentiate between these categories by analyzing a set of pre-classified emails. The goal is to apply this learned understanding to classify new, unseen emails accurately.

# How It Works
# Phase 1: Training
During the training phase, the classifier processes two separate datasets: one containing spam emails and the other containing not-spam emails. It extracts and analyzes the words found in these emails, noting the presence or absence of each word without considering their frequency. This phase aims to build a model that understands the characteristics distinguishing spam from not-spam emails based on the training data provided.

# Phase 2: Testing
In the testing phase, the classifier is presented with a new set of emails. It applies the Naive Bayes algorithm to each email, calculating the probability of it being spam or not-spam based on the features observed during training. The classifier then predicts the category of each email, providing a detailed log of its predictions, including the calculated probabilities and the final classification.

# Features and Implementation Details
Features Used: The presence or absence of every word observed in the training set. The classifier does not consider the frequency of words within an email.
Probability Calculation: To avoid numerical underflow, the classifier uses log-probabilities.
Input Format: The program expects four text files containing the training and testing datasets for both spam and not-spam emails. Each file follows a specific format, detailing the subject and body of each email.
Output
For every email in the test set, the classifier outputs:

A line indicating the test email number, the number of relevant features found, the calculated log-probabilities for both classes, the predicted class, and whether the prediction was accurate.
A summary at the end of the output detailing the total number of emails correctly classified.

# Usage
The program prompts the user to input the filenames of the training and testing datasets for both spam and not-spam emails. After processing, it provides a detailed classification report for each email in the testing set, alongside a summary of its overall accuracy.

# Objective
The purpose of this project is to demonstrate the practical application of the Naive Bayes classifier in distinguishing between spam and not-spam emails. It showcases the classifier's ability to learn from a given dataset and apply this knowledge to make predictions on new data. While perfection in classification accuracy is not expected due to the complexity of language and email content, the project aims to achieve a reasonable level of accuracy in email classification.

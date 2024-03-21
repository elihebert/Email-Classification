import math
from collections import defaultdict
import re

#function to clean and split text into words
def clean_and_split(text):
    #convert to lower case and split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    return words

#function to process email content/extract features
#def process_emails(file_content):    #split content into emails using subject and body tags as delimiters
#    emails = re.split(r'<SUBJECT>|<BODY>', file_content)[1:] #split and remove first empty result
    #remove last </BODY> tag (if it exists) and other extra whitespace
#    emails = [email.replace('</SUBJECT>\n', '').replace('</BODY>\n', '').replace('</BODY>', '').strip() for email in emails if email.strip()]
#
#    #creation of list to hold tuples of subject and body words
#    processed_emails = []
#    for i in range(0, len(emails), 2):
#        subject, body = emails[i:i+2] #get subject and body pair of email
#
        #clean and split subject and body into words
#       subject_words = set(clean_and_split(subject))
#       body_words = set(clean_and_split(body))
#
#       #combine words from both subject and body, maintaining uniqueness
#       combined_words = subject_words.union(body_words)
#       processed_emails.append(combined_words)
#    return processed_emails

#also a function to process email content/extract features; this one uses regex to split content into pairs of subject and body,
#and then uses zip to create pairs of subject and body, using set union to combine words from both subject and body; this
#function seems much more efficient than the one above, however the one above is slighty more accurate
EMAIL_PARTS = ('<SUBJECT>', '<BODY>')
def process_emails(file_content):
    # Use regex to split content into pairs of subject and body
    parts = re.split('|'.join(EMAIL_PARTS), file_content)[1:]  # Skip the first empty result
    emails = zip(*[iter(parts)]*2)  # Create pairs of subject and body

    # Clean and combine words from both subject and body
    return [set(clean_and_split(sub)) | set(clean_and_split(body)) for sub, body in emails]


#function to read and preprocess training data, then calculcate prior probabilities and likelihoods
def train_naive_bayes(train_spam, train_ham):
    #process training files and create vocabulary and word counts
    processed_spam = process_emails(train_spam)
    processed_ham = process_emails(train_ham)

    #calculate vocabulary size, and counts of words in spam and ham
    vocabulary = set()
    spam_word_counts = defaultdict(int)
    ham_word_counts = defaultdict(int)

    #process spam emails to update spam word counts and vocabulary
    for email in processed_spam:
        for word in email:
            vocabulary.add(word)
            spam_word_counts[word] += 1

    #process ham emails to update ham word counts and vocabulary
    for email in processed_ham:
        for word in email:
            vocabulary.add(word)
            ham_word_counts[word] += 1

    #calculate prior probabilities
    total_emails = len(processed_spam) + len(processed_ham)
    prior_spam = len(processed_spam) / total_emails
    prior_ham = len(processed_ham) / total_emails

    #calculate likelihoods
    spam_likelihoods = defaultdict(float)
    ham_likelihoods = defaultdict(float)

    for word in vocabulary:
        #laplace smoothing is applied; +1 for numerator and +len(vocabulary) for denominator
        spam_likelihoods[word] = (spam_word_counts[word] + 1) / (len(processed_spam) + 2)
        ham_likelihoods[word] = (ham_word_counts[word] + 1) / (len(processed_ham) + 2)

    return prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary

#function to classify a single email
def classify_email(email, prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary):
    #initialize log probabilities to prior probabilities
    log_prob_spam = math.log(prior_spam)
    log_prob_ham = math.log(prior_ham)

    #add log likelihoods for each word in email
    for word in email:
        if word in vocabulary: #ignore words not in vocabulary
            log_prob_spam += math.log(spam_likelihoods[word])
            log_prob_ham += math.log(ham_likelihoods[word])

        #note that if a word is not in the training set, it does not contribute to the score

    for word in vocabulary:
        if word not in email:
            log_prob_spam += math.log(1 - spam_likelihoods[word])
            log_prob_ham += math.log(1 - ham_likelihoods[word])

    #return calculcated log probabilities
    return log_prob_spam, log_prob_ham

def test_naive_bayes(test_spam, test_ham, prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary):
    #process test files
    processed_test_spam = process_emails(test_spam)
    processed_test_ham = process_emails(test_ham)

    #initialize counters for correct classifications
    correct_spam = 0
    correct_ham = 0
    spam_email_count = len(processed_test_spam)
    ham_email_count = len(processed_test_ham)

    #classify each spam email and check if the classification is correct
    for i, email in enumerate(processed_test_spam, 1):
        log_prob_spam, log_prob_ham = classify_email(email, prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary)
        classification = "spam" if log_prob_spam > log_prob_ham else "ham"
        correctness = "right" if classification == "spam" else "wrong"
        if correctness == "right":
            correct_spam += 1
        junction = len(email.intersection(vocabulary))

        #print formatted output
        print(f"TEST {i} {junction}/{len(vocabulary)} features true {log_prob_spam:.3f} {log_prob_ham:.3f} {classification} {correctness}")

    #classify each ham email and check if the classification is correct
    for i, email in enumerate(processed_test_ham, 1):
        log_prob_spam, log_prob_ham = classify_email(email, prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary)
        classification = "spam" if log_prob_spam > log_prob_ham else "ham"
        correctness = "right" if classification == "ham" else "wrong"
        if correctness == "right":
            correct_ham += 1
        junction = len(email.intersection(vocabulary))

        #print formatted output
        print(f"TEST {i} {junction}/{len(vocabulary)} features true {log_prob_spam:.3f} {log_prob_ham:.3f} {classification} {correctness}")

    #print the total number of correctly classified emails
    total_correct = correct_spam + correct_ham
    total_emails = spam_email_count + ham_email_count
    print(f"Total: {total_correct}/{total_emails} emails classified correctly.")

if __name__ == "__main__":
    # Prompt for file names
    train_spam_filename = input("Enter the training file for spam: ")
    train_ham_filename = input("Enter the training file for ham: ")
    test_spam_filename = input("Enter the testing file for spam: ")
    test_ham_filename = input("Enter the testing file for ham: ")
    
    # Read file contents
    with open(train_spam_filename, 'r') as file:
        train_spam = file.read()
    with open(train_ham_filename, 'r') as file:
        train_ham = file.read()
    with open(test_spam_filename, 'r') as file:
        test_spam = file.read()
    with open(test_ham_filename, 'r') as file:
        test_ham = file.read()

    # Train the Naive Bayes classifier
    prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary = train_naive_bayes(train_spam, train_ham)

    # Test the Naive Bayes classifier
    test_naive_bayes(test_spam, test_ham, prior_spam, prior_ham, spam_likelihoods, ham_likelihoods, vocabulary)
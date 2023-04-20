import ahocorasick

# Load Swahili dictionary
with open('train.csv', 'r', encoding='utf-8') as f:
    swahili_dict = set([line.strip() for line in f.readlines()])

# Build Aho-Corasick automaton for dictionary matching
AC = ahocorasick.Automaton()
for word in swahili_dict:
    AC.add_word(word, word)
AC.make_automaton()

# Define a function to perform spelling correction
def correct_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        if word in swahili_dict:
            corrected_words.append(word)
        else:
            for _, match in AC.iter(word):
                corrected_words.append(match)
                break
    return ' '.join(corrected_words)

# Example usage
text = "Habari wazungu, mbona hamjapanda juu ya ndege?"
corrected_text = correct_spelling(text)
print(corrected_text)
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load Swahili sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the CSV file
df = pd.read_csv('train.csv')

# Iterate over each row of the dataframe and analyze the sentiment
for i, row in df.iterrows():
    sentiment = sia.polarity_scores(row['content'])['compound']
    if sentiment > 0.5:
        df.loc[i, 'Sentiment'] = 'Positive'
    elif sentiment < -0.5:
        df.loc[i, 'Sentiment'] = 'Negative'
    else:
        df.loc[i, 'Sentiment'] = 'Neutral'

# Save the results to a new CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
from googletrans import Translator

# Create an instance of the Translator class
translator = Translator(service_urls=['translate.google.com'])

# Define a function to translate text from Swahili to English
def swahili_to_english(text):
    translation = translator.translate(text, src='sw', dest='en')
    return translation.text

# Define a function to translate text from English to Swahili
def english_to_swahili(text):
    translation = translator.translate(text, src='en', dest='sw')
    return translation.text

# Example usage
print(swahili_to_english('Habari za asubuhi?'))  # prints 'Good morning?'
print(english_to_swahili('How are you?'))  # prints 'Vipi hali yako?'
import kraken

# Load the pre-trained model
model = kraken.Model.load('kraken/models/kan_lev1.mlmodel')

# Read in the handwritten document
with open('handwritten_doc.jpg', 'rb') as f:
    img_bytes = f.read()

# Recognize the text in the image
text = model.recognize(img_bytes)

# Print the recognized text
print(text)
from transformers import pipeline

# Create a question answering pipeline
qa_pipeline = pipeline("question-answering", model="t5-base", tokenizer="t5-base")

# Define function for question answering
def ask_question(context, question):
    input_text = "swahili question: {} context: {}".format(question, context)
    output = qa_pipeline(input_text)[0]
    answer = output['answer']
    return answer

# Define some context for the chatbot to use
context = "Mwalimu aliingia darasani na kuanza somo la hesabu."

# Loop for chatbot interaction
print("Karibu kwenye mazungumzo! Unaweza kuniuliza maswali yoyote kuhusu hii hadithi.")
while True:
    # Get user input
    user_input = input("Nini unataka kujua? (au andika 'kwa heri' kuondoka) ")
    
    # Check for exit command
    if user_input.lower() == 'kwa heri':
        print("Asante kwa mazungumzo, tutaonana baadaye!")
        break
    
    # Answer user's question
    answer = ask_question(context, user_input)
    print("Jibu: ", answer)

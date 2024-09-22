import nltk
import random 
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore') 


file = open('culturalimpact.txt', 'r', errors='ignore')
text = file.read().lower()
nltk.download('punkt_tab')
nltk.download('wordnet')

sent_list = nltk.sent_tokenize(text)  
word_list = nltk.word_tokenize(text)  


first_sent = sent_list[:4]
first_word = word_list[:4]


lem = nltk.stem.WordNetLemmatizer()


def lemmatize_words(words):
    return [lem.lemmatize(word) for word in words]


punct_remove = dict((ord(p), None) for p in string.punctuation)


def normalize_text(text):
    return lemmatize_words(nltk.word_tokenize(text.lower().translate(punct_remove)))


greetings = ("hello", "hi", "greetings", "sup", "what's up", "hey")
greet_responses = ["hi", "hey", "hello", "hi there", "glad to talk!"]


def check_greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetings:
            return random.choice(greet_responses)



def bot_reply(user_input):
    bot_resp = ''
    sent_list.append(user_input)  
    vec = TfidfVectorizer(tokenizer=normalize_text, stop_words="english")
    tfidf = vec.fit_transform(sent_list)  
    sim_vals = cosine_similarity(tfidf[-1], tfidf)  
    idx = sim_vals.argsort()[0][-2]  
    flat_vals = sim_vals.flatten()  
    flat_vals.sort()
    sim_score = flat_vals[-2]

    if sim_score == 0:  
        bot_resp += "Sorry, I don't understand."
        return bot_resp
    else:  
        bot_resp += sent_list[idx]
        return bot_resp


if __name__== "__main__":
    active = True
    print("Hi, I'm Bot! Ask me anything. Type 'bye' to exit.")

    while active:
        user_input = input().lower() 
        if user_input != 'bye':  
            if user_input == 'thanks' or user_input == 'thank you':
                active = False
                print("Bot: You're welcome!")
            else:
                if check_greeting(user_input) is not None: 
                    print("Bot:", check_greeting(user_input))
                else: 
                    print("Bot:", bot_reply(user_input))
                    sent_list.remove(user_input)  
        else:
            active = False
            print("Bot: Bye! Take care!")

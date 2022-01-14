import aiml
import requests
from nltk import *
import pandas as pd
from nltk.corpus import wordnet as wn
import string
from nltk.inference import ResolutionProver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import numpy as np
import time
import speech_recognition as sr
from gtts import gTTS
import playsound

read_expr = logic.Expression.fromstring

#######################################################
#  Initialise Knowledgebase(s) and check the consistency
#######################################################

kb = []
df = pd.read_csv("kb_final.csv", header=None)

print("Please wait a moment while the integrity of the KB file checked...") 

[kb.append(read_expr(row)) for row in df[0]]

start = time.time()
#measuring time of Resolution Prover vs Prover9 
is_valid = ResolutionProver().prove(None, kb, verbose=False)
if is_valid:
   sys.exit("The Knowledgebase is not consistent - Please remove any contradictions and run this program again.")
"""
if Prover9Command(assumptions=kb).prove():
   sys.exit("The Knowledgebase is not consistent - Please remove any contradictions and run this program again.")
"""
end = time.time()
print("Time taken: " + str(round(end - start, 3)) + " seconds") 

food_groups =  ["food", "protein", "vegetable", "fruit", "carbohydrate", "dairy", "fat", "meat"]
synset1 = wn.synset('food.n.01')
synset2 = wn.synset('food.n.02')
common_foods = list(set([w for s in synset1.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
common_foods += list(set([w for s in synset2.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
alphabet_list = list(string.ascii_uppercase)
common_foods = [item for item in common_foods if item not in alphabet_list]
#######################################################
#  Initialise QA Pairs
#######################################################
qa = pd.read_csv("QA.csv").dropna()
#######################################################
#  Initialise AIML agent
#######################################################
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel"s bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
kern.bootstrap(learnFiles="mybot-logic.xml")
#######################################################
#  Initialise Nutrition API URL
#######################################################
url = "https://api.edamam.com/api/nutrition-data?app_id=a4db010c&app_key=%20059652af8242eb8f8c2d06ad34b95437&nutrition-type=logging&ingr="
nutrients = {"calories": "ENERC_KCAL", 
     "carbs" : "CHOCDF", 
     "carbohydrates": "CHOCDF",
     "fat": "FATS", 
     "fiber" : "FIBTG", 
     "protein": "PROCNT"}
#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")
#######################################################
# Main loop
#######################################################
   
def validate_expression(object, subject):
    expr = read_expr((subject + "(" + object + ")").lower())
    print(expr)
    start = time.time()
    is_valid_expression = ResolutionProver().prove(expr, kb, verbose=False)
    end = time.time()
    print("Time taken: " + str(round(end - start, 3)) + " seconds") 
    return is_valid_expression
"""
def validate_expression(object, subject):
    expr = read_expr((subject + "(" + object + ")").lower())
    print(expr)
    start = time.time()
    is_valid_expression = Prover9Command(expr, assumptions=kb).prove()
    end = time.time()
    print("Time taken: " + str(round(end - start, 3)) + " seconds") 
    return is_valid_expression
"""
def get_fuzzy_match(word, data):
    choices = []
    for key in data:
        choices.append(key)
    result, score = process.extractOne(word, choices)
    if score > 80:
        if word.find("not ") != -1:
            return "not " + result
        else:
            return result
    else:
        return word

#audio communication    
r = sr.Recognizer()

def get_audio():
    with sr.Microphone() as source:
        try:
            # read the audio data from the default microphone
            audio_data = r.record(source, duration=5)
            print("Recognizing...")
            # convert speech to text
            text = r.recognize_google(audio_data, language="en-US")
            return(text)
        except:
            return("unrecognisable")

def print_with_audio(input_type, robot_input):
    if input_type == 'text':
        print(robot_input)
    else:
        print(robot_input)
        tts = gTTS(text=robot_input, lang='en')
        tts.save("tts.mp3")
        playsound.playsound("tts.mp3")

input_type = input("Choose 'text' or 'speech' based communication> ")

while True:
    # get user input
    try:
        if input_type == 'text':
            user_input = input("> ")
        else:
            user_input = get_audio()
            print("> " + user_input)
            
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    answer = kern.respond(user_input)
    # post-process the answer for commands
    if answer[0] == "#":
        params = answer[1:].split("$")
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        # Here are the processing of the new logical component:
        # Task B
        elif cmd == 31:  # if input pattern is "I know that * is *"
            object, subject = params[1].split(" is ")
            object = get_fuzzy_match(object, common_foods)
            subject = get_fuzzy_match(subject, food_groups)
            if validate_expression(object, subject):
                print_with_audio(input_type, "I already knew that")
            else:
                #Check reverse expression to prove if there is an rule in knowledgebase (or not) that disproved original expression
                if validate_expression(object, ("not " + subject)):
                    print_with_audio(input_type, "I can see that is NOT true")
                else:
                    print_with_audio(input_type, ("OK, I will remember that " + object + " is " + subject))
                    kb.append(read_expr(subject + "(" + object + ")"))
        elif cmd == 32:  # if the input pattern is "check that * is *"
            object, subject = params[1].split(" is ")
            object = get_fuzzy_match(object, common_foods)
            subject = get_fuzzy_match(subject, food_groups)
            if validate_expression(object, subject):
                print_with_audio(input_type, "Correct")
            else:
                #Check reverse expression to prove if there is an rule in knowledgebase (or not) that disproved original expression
                if validate_expression(object, ("not " + subject)):
                    print_with_audio(input_type, "Incorrect")
                else:
                    print_with_audio(input_type, "I'm not sure...")
        elif cmd == 33: # if input pattern is "I know that * and * are a pair"
            first, second = params[1].split(" and ")
            first = get_fuzzy_match(first, common_foods)
            second = get_fuzzy_match(second, common_foods)
            objects = first + "," + second
            if (validate_expression(first, "food") and validate_expression(second, "food")):
                if validate_expression(objects, "not combo"):
                    print_with_audio(input_type, "I can see that is NOT a tasty combo!")
                else:
                    if validate_expression(objects, "pair"):
                        print_with_audio(input_type, ("I already know that " + params[1] + " are a classic combination."))
                    else:
                        kb.append(read_expr("food(" + first + ") & food(" + second + ")"))
                        kb.append(read_expr("combo(" + objects + ")"))
                        print_with_audio(input_type, "That new combo has been added!")
            else:
                print_with_audio(input_type, "Please use the input pattern 'I know that * is *' - To categorise these items as food before making them a pair.")
        elif cmd == 40: # if input pattern is "how much * is in *" or "how many * are in *"
           nutrient, food = params[1].split(" is ")
           label = get_fuzzy_match(nutrient, nutrients)
           food = get_fuzzy_match(food, common_foods)
           key = nutrients.get(label)
           response = requests.get(url + food.replace("_", "+").replace("-", "+"))
           try:
               nutrient_data = response.json()["totalNutrients"][key]
               food = food.replace("_", " ").replace("-", " ")
               print_with_audio(input_type, ("Amount of " + label +  " in " + food + " is: " + str(round(nutrient_data["quantity"])) + nutrient_data["unit"]))
           except KeyError:
               print("That food was not found. Check spelling.")
        elif cmd == 41: # if input pattern is "what is the full nutrition for *" or "nutrition for *"
           food = user_input.split("for ",1)[1]
           food = get_fuzzy_match(food, common_foods)
           response = requests.get(url + food.replace("_", "+").replace("-", "+"))
           nutrient_data = response.json()
           percentages = []
           labels = []
           
           for i in nutrient_data["totalDaily"]:
               percentages.append(nutrient_data["totalDaily"][i]["quantity"])
               labels.append(nutrient_data["totalDaily"][i]["label"])
           
           plt.xticks(range(len(percentages)), labels)
           plt.xticks(rotation=90)
           
           plt.xlabel('Nutrient')
           plt.ylabel('Percentage (%)')
           food = food.replace("_", " ").replace("-", " ")
           plt.title('Nutrients percentage against the recommended daily for ' + food)
           plt.bar(range(len(percentages)), percentages)
           plt.show()
           
        # Task A
        elif cmd == 99: #if nothing matches from aiml, then default to this section which is QA file.
            qa_questions = qa["questions"]
            qa_answers = qa["answers"]
    
            questions = [user_input]
            answers = []

            for i in range(len(qa_questions)):
                questions.append(qa_questions[i])
                answers.append(qa_answers[i])

            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(questions)
            
            j = 0
            max_score = 0
            
            for i in range(1, len(questions)):
                cosine_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[i])[0][0]
                if cosine_score > max_score:
                    max_score = cosine_score
                    j = i          
            if max_score == 0:
                try:
                    fuzzy_question = get_fuzzy_match(user_input,qa_questions)
                    i = questions.index(fuzzy_question)
                    print_with_audio(input_type, answers[i-1])
                except (IndexError, ValueError) as e:
                    print_with_audio(input_type, "Hey I don't understand what you just wrote to me...")
            else:
                print_with_audio(input_type, answers[j-1])
            
    else:
        print_with_audio(input_type, answer)
        
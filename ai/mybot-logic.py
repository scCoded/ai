#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications - rework task A
"""
import aiml
import requests
from nltk.sem.logic import *
from nltk.inference.resolution import *
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import nltk.inference.resolution as res
from nltk.inference.nonmonotonic import *
from nltk.inference.api import *
from nltk import *
from nltk.sem import logic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

read_expr = logic.Expression.fromstring


#######################################################
#  Initialise Knowledgebase(s) and check the consistency
#######################################################
kb = []
kb2 = []

kb3 = []
df = pd.read_csv("kb_final.csv", header=None)
data = pd.read_csv("SampleQA.csv").dropna()
data.head()
[kb3.append(read_expr(row)) for row in df[0]]

#for i in range(0,len(kb3)):
#    expr = kb3[i] 
#    tempKb = kb3[:]
#    tempKb.pop(i)
#    print(expr)
   # print(ResolutionProver().prove(expr, tempKb, verbose=False))

#def load_and_check_kb(file_name, knowledgebase):
#    with open(file_name) as file:
#        lines = (line.rstrip() for line in file)
#        lines = (line for line in lines if line)
#        for line in lines:
#            knowledgebase.append((read_expr(line)))
#    for expr in knowledgebase:
         
#        if not is_consistent:
#           sys.exit("The Knowledgebase is not consistent - Please remove any contradictions and run this program again.")


print("Please wait a moment while the integrity of the KB file checked...")
#load_and_check_kb("kb.csv", kb)    
#load_and_check_kb("kb_extra.csv", kb2)

if Mace(end_size=50).build_model(None, kb3) == False :
   sys.exit("The Knowledgebase is not consistent - Please remove any contradictions and run this program again.")

food_groups =  ["food", "protein", "vegetable", "fruit", "carbohydrate", "dairy", "fat", "meat"]
#######################################################
#  Initialise QA Pairs
#######################################################
qa = pd.read_csv("sampleQA.csv").dropna()
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
    
def validate_expression(knowledgebase, object, subject):
    expr = read_expr((subject + "(" + object + ")").lower())
    print(expr)
    is_valid_expression = Prover9Command(expr, assumptions=knowledgebase).prove()
    #is_valid_expression = ResolutionProver().prove(expr, knowledgebase, verbose=False)
    return is_valid_expression

def get_fuzzy_match(word, data):
    choices = []
    for key in data:
        choices.append(key)
    result = process.extractOne(word, choices)
    if word.find("not ") != -1:
        return "not " + result[0]
    else:
        return result[0]

while True:
    # get user input
    try:
        user_input = input("> ")
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
            subject = get_fuzzy_match(subject, food_groups)
            if validate_expression(kb3, object, subject):
                print("I already knew that")
            else:
                #Check reverse expression to prove if there is an rule in knowledgebase (or not) that disproved original expression
                if validate_expression(kb3, object, ("not " + subject)):
                    print("I can see that is not true")
                else:
                    print("OK, I will remember that", object, "is a", subject)
                    kb.append(read_expr(subject + "(" + object + ")"))
        elif cmd == 32:  # if the input pattern is "check that * is *"
            object, subject = params[1].split(" is ")
            subject = get_fuzzy_match(subject, food_groups)
            if validate_expression(kb3, object, subject):
                print("Correct")
            else:
                #Check reverse expression to prove if there is an rule in knowledgebase (or not) that disproved original expression
                if validate_expression(kb3, object, ("not " + subject)):
                    print("Incorrect")
                else:
                    print("I'm not sure...")
        elif cmd == 33: # if input pattern is "I know that * and * are a pair"
            first, second = params[1].split(" and ")
            objects = first + "," + second
            #if validate_expression(kb3, objects, "pair"):
            text_expr = read_expr("food(" + first + ")")
            print("here")
            print(ResolutionProverCommand(text_expr, kb3).prove())
            if validate_expression(kb3, first, "food") and validate_expression(kb3, second, "food"):
                expr = read_expr(("pair" + "(" + objects + ")").lower())
                if Mace(end_size=50).build_model(expr, kb3) == False:
                    print("I already know that " + params[1] + " are a classic combination.")
                else:
                    print("That new combo has been added!")
                    kb3.append(read_expr("food(" + first + ") & food(" + second + ")"))
                    kb3.append(read_expr("combo(" + objects + ")"))
            else:
               print("Please use the input pattern 'I know that * is *' - To categorise these items as food before making them a pair.")
        elif cmd == 40: # if input pattern is "how much * is in *" or "how many * are in *"
            nutrient, food = params[1].split(" is ")
            label = get_fuzzy_match(nutrient, nutrients)
            key = nutrients.get(label)
            response = requests.get(url + food.replace(" ", "+"))
            try:
                nutrient_data = response.json()["totalNutrients"][key]
                print("Amount of " + label +  " in " + food + " is: " + str(round(nutrient_data["quantity"])) + nutrient_data["unit"])
            except KeyError:
                print("That food was not found. Check spelling.")
        elif cmd == 41: # if input pattern is "what is the full nutrition for *" or "nutrition for *"
           food = user_input.split("for ",1)[1]
           response = requests.get(url + food.replace(" ", "+"))
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
                print("Hey I don't understand what you just wrote to me")
            else:
                print(answers[j-1])
            

    else:
        print(answer)
from collections import Counter

file_path = "cleaned_merged_fairy_tales_without_eos.txt"


def blank_lines():
    with open(file_path, "r") as f:
        lines = f.readlines()
        empty = 0 
        for line in lines:
            # check if line is empty
            if line == '\n':
                empty += 1 

    return empty

def characters():
    with open(file_path, "r") as f:
        lines = f.readlines()
        characters = 0 
        for line in lines:
            characters += len(line) 

    return characters
        

def words():
    with open(file_path, "r") as f:
        lines = f.readlines()
        words = 0 
        for line in lines:
            words += len(line.split()) 

    return words

def sentences():
    with open(file_path, "r") as f:
        lines = f.readlines()
        sentences = 0 
        for line in lines:
            sentences += len(line.split(".")) 

    return sentences

# unique words 
def unique_words():
    with open(file_path, "r") as f:
        lines = f.readlines()
        unique_words = {} 
        for line in lines:
            for word in line.split():
                unique_words[word] = unique_words.get(word, 0) + 1
        

    return unique_words

def count_punctuation():
    with open(file_path, "r") as f:
        question_marks = 0 
        exclamation_marks = 0
        periods = 0
        commas = 0
        lines = f.readlines()
        for line in lines:
            for char in line:
                if char == "?":
                    question_marks += 1
                elif char == "!":
                    exclamation_marks += 1
                elif char == ".":
                    periods += 1
                elif char == ",":
                    commas += 1
    return question_marks, exclamation_marks, periods, commas


print("number of lines with all caps: ", all_caps_line())

# print("Number of blank lines: ", blank_lines())
# print("Number of characters: ", characters())
# print("Number of words: ", words())
# print("Number of sentences: ", sentences())
# print("Number of unique words: ", len(unique_words()))
# # most common 10 words and counts
# print("Most common 10 words and counts: ", Counter(unique_words()).most_common(40))

# print("Number of question marks: ", count_punctuation()[0])
# print("Number of exclamation marks: ", count_punctuation()[1])
# print("Number of periods: ", count_punctuation()[2])
# print("Number of commas: ", count_punctuation()[3])

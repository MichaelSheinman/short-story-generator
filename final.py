# import the lstm model
from lstm_model import *
# import the story generator model
from story_generator import *
non_info_words = ["a", "the", "or", "he", "she", "in", "on", "my", "it"]

# importing timer to create typewriter effect
from time import sleep

def typewriter_effect(words, speed):
    for char in words:
        sleep(speed)
        print(char, end='', flush=True)

def generate_story(prompt):
    completed_story = complete_prompt(prompt, model2)
    word_occurences = {}
    # get the words that occur the most often in the dictionary
    for word in completed_story:
        if not (word.lower() in non_info_words):
            if word not in word_occurences:
                word_occurences[word] = 1
            else:
                word_occurences[word] += 1

    # Get the words from most to least occurrences
    # [:3] gets the top 3 most popular words
    story_vocab = sorted(word_occurences)[::-1][:3]

    # The first word in the title will be the word that occurs the most often in the story
    final_title = []
    final_title.append(story_vocab[0])
    # get generated title and include it to the final title
    generated_title = generate_title()
    final_title.extend(generated_title)
    final_title.extend(story_vocab[1:])
    final_title = ' '.join(final_title).upper()
    # return the completed title and the story
    typewriter_effect(final_title, 0.1)
    typewriter_effect(completed_story, 0.1)

if __name__ == "__main__":
    print("------------ HELLO! I am your personal bedtime story assitant ------------\n")
    print("--------------------------------------------------------\n")
    prompt = input('Please write the beginning of your bedtime story and we will help you write the rest: \n')
    print("--------------------------------------------------------\n")
    typewriter_effect("COMPLETING STORY", 0.1)
    typewriter_effect("... ...  ...        ... ...  ...\n", 0.2)
    typewriter_effect("GENERATING TITLE", 0.1)
    typewriter_effect("... ...  ...        ... ...  ...\n", 0.2)
    print("--------------------------------------------------------\n")
    print("--------------------------------------------------------\n")
    generate_story(prompt)
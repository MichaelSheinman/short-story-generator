# import the lstm model
from lstm_model import *
# import the story generator model
from story_generator import *
# stop words that don't add context to title
non_info_words = ['and', 'able', 'about', 'abroad', 'according', 'accordingly', 'across', 'actually', 'adj',
              'after', 'afterwards', 'again', 'ahead', "ain't", 'all', 'allow', 'allows', 'almost', 'along', 'alongside',
              'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'an', 'and', 'another',
              'any', 'anyhow', 'anything', 'anyway', 'anyways', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around',
              'as', "a's", 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'back', 'backward',
              'backwards', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin',
              'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both',
              'brief', 'but', 'by', 'came', 'can', 'cannot', 'cant', "can't", 'caption', 'cause', 'causes', 'certain',
              'certainly', 'changes', 'clearly', "c'mon", 'co', 'co.', 'com', 'come', 'comes', 'concerning', 'consequently',
              'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course',
              "c's", 'currently', 'dare', "daren't", 'definitely', 'described', 'despite', 'did', "didn't", 'different',
              'directly', 'do', 'does', "doesn't", 'doing', 'done', "don't", 'down', 'downwards', 'during', 'each', 'edu',
              'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'entirely', 'especially',
              'et', 'etc', 'even', 'ever', 'evermore', 'every', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except',
              'fairly', 'few', 'fewer', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'forever', 'formerly',
              'forth', 'forward', 'found', 'four', 'from', 'further', 'furthermore', 'get', 'gets', 'getting', 'given', 'gives',
              'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'had', "hadn't", 'half', 'happens', 'hardly', 'has',
              "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", 'hello', 'help', 'hence', 'her', 'here', 'hereafter',
              'hereby', 'herein', "here's", 'hereupon', 'hers', 'herself', "he's", 'hi', 'him', 'his', 'hither', 'hopefully', 'how',
              'howbeit', 'however', 'hundred', "i'd", 'ie', 'if', 'ignored', "i'll", "i'm", 'immediate', 'in', 'inasmuch', 'inc',
              'inc.', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'inside', 'insofar', 'instead', 'into', 'inward',
              'is', "isn't", 'it', "it'd", "it'll", 'its', "it's", 'itself', "i've", 'just', 'k', 'keep', 'keeps', 'kept', 'know',
              'known', 'knows', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked',
              'likely', 'likewise', 'little', 'look', 'looking', 'looks', 'low', 'lower', 'ltd', 'made', 'mainly', 'make', 'makes',
              'many', 'may', 'maybe', "mayn't", 'me', 'mean', 'meantime', 'meanwhile', 'merely', 'might', "mightn't", 'mine',
              'minus', 'miss', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', "mustn't", 'my', 'myself',
              'name', 'namely', 'nd', 'nearly', 'necessary', 'need', "needn't", 'needs', 'neither', 'never', 'neverf', 'neverless',
              'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'non', 'none', 'nonetheless', 'noone', 'no-one', 'nor',
              'normally', 'not', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'obviously', 'of', 'off', 'often', 'oh',
              'ok', 'okay', 'on', 'once', 'one', 'ones', "one's", 'onto', 'opposite', 'or', 'other', 'others', 'otherwise', 'ought',
              "oughtn't", 'our', 'ours', 'ourselves', 'out', 'over', 'overall', 'own', 'particular', 'particularly', 'past', 'per',
              'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provided', 'provides', 'que', 'quite',
              'qv', 'rather', 'rd', 're', 'really', 'reasonably', 'recent', 'recently', 'regarding', 'regardless', 'regards',
              'relatively', 'respectively', 'right', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'seem',
              'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several',
              'shall', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody',
              'someday', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry',
              'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 'take', 'taken', 'taking', 'tell',
              'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that's", "that've", 'the', 'their',
              'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', "there'd", 'therefore', 'therein',
              "there'll", "there're", 'theres', "there's", 'thereupon', "there've", 'these', 'they', "they'd", "they'll", "they're",
              "they've", 'thing', 'things', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three',
              'through', 'throughout', 'thru', 'thus', 'till', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried',
              'tries', 'truly', 'try', 'trying', "t's", 'twice', 'two', 'un', 'under', 'underneath', 'undoing', 'unfortunately',
              'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'upwards', 'us', 'use', 'used', 'useful', 'uses',
              'using', 'usually', 'v', 'value', 'various', 'versus', 'very', 'via', 'viz', 'vs', 'want', 'wants', 'was', "wasn't",
              'way', 'we', "we'd", 'welcome', 'well', "we'll", 'went', 'were', "we're", "weren't", "we've", 'what', 'whatever',
              "what'll", "what's", "what've", 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein',
              "where's", 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whither', 'who', "who'd",
              'whoever', 'whole', "who'll", 'whom', 'whomever', "who's", 'whose', 'why', 'will', 'willing', 'wish', 'with',
              'within', 'without', 'wonder', "won't", 'would', "wouldn't", 'yes', 'yet', 'you', "you'd", "you'll", 'your',
              "you're", 'yours', 'yourself', 'yourselves', "you've", 'zero', 'a', "how's", 'i', "when's", "why's", 'b', 'c',
              'd', 'e', 'f', 'g', 'h', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'uucp', 'w', 'x', 'y', 'z', 'I',
              'www', 'amount', 'bill', 'bottom', 'call', 'computer', 'con', 'couldnt', 'cry', 'de', 'describe', 'detail', 'due',
              'eleven', 'empty', 'fifteen', 'fifty', 'fill', 'find', 'forty', 'front', 'full', 'give', 'hasnt', 'herse', 'himse',
              'interest', 'itse”', 'mill', 'myse”', 'part', 'put', 'side', 'sincere', 'sixty', 'system', 'ten', 'thick', 'thin',
              'top', 'twelve', 'twenty', 'abst', 'accordance', 'act', 'added', 'affected', 'affecting', 'affects', 'ah', 'announce',
              'anymore', 'apparently', 'approximately', 'aren', 'arent', 'arise', 'auth', 'beginning', 'beginnings', 'begins',
              'biol', 'briefly', 'ca', 'date', 'ed', 'effect', 'et-al', 'ff', 'fix', 'gave', 'giving', 'heres', 'hes', 'hid', 'id',
              'im', 'immediately', 'importance', 'important', 'index', 'information', 'itd', 'keys', 'kg', 'km', 'largely', 'lets',
              'line', "'ll", 'means', 'mg', 'million', 'ml', 'mug', 'na', 'nay', 'necessarily', 'nos', 'noted', 'obtain',
              'obtained', 'omitted', 'ord', 'owing', 'page', 'pages', 'poorly', 'possibly', 'potentially', 'pp', 'predominantly',
              'previously', 'primarily', 'promptly', 'proud', 'readily', 'related', 'research', 'resulted', 'resulting', 'results',
              'sec', 'section', 'shed', 'shes', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 'similar',
              'similarly', 'slightly', 'somethan', 'specifically', 'state', 'states', 'stop', 'strongly', 'substantially',
              'successfully', 'sufficiently', 'suggest', 'thered', 'thereof', 'therere', 'thereto', 'theyd', 'theyre', 'thou',
              'thoughh', 'thousand', 'throug', 'til', 'tip', 'ts', 'ups', 'usefully', 'usefulness', "'ve", 'vol', 'vols', 'wed',
              'whats', 'wheres', 'whim', 'whod', 'whos', 'widely', 'words', 'youd', 'youre']

# importing timer to create typewriter effect
from time import sleep

def typewriter_effect(words, speed):
    for char in words:
        sleep(speed)
        print(char, end='', flush=True)

from string import punctuation # to format the story later ...

def generate_story(prompt):
    completed_story = complete_prompt(prompt)
    r = re.compile(r'[{}]+'.format(re.escape(punctuation)))
    story_just_words = r.sub('', completed_story)
    story_just_words = story_just_words.lower().split(" ")
    word_occurences = {}
    # get the words that occur the most often in the dictionary
    for word in  story_just_words:
        if not (word in non_info_words):
            if word not in word_occurences:
                word_occurences[word] = 1
            else:
                word_occurences[word] += 1

    # sorting the dictionary from least to most occurences
    word_occurences = {k: v for k, v in sorted(word_occurences.items(), key=lambda item: item[1])}

    # Get the words from most to least occurrences
    story_vocab = list(word_occurences.keys())[::-1]

    i = 0
    first_word = story_vocab[i]
    while first_word not in VOCABULARY:
        first_word = story_vocab[i + 1]

    # The first word in the title will be the word that occurs the most often in the story
    # get generated title and include it to the final title
    final_title = generate_title(first_word)
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
# CODE TO PREPROCESS THE DATA -------------
import re

data_file = "cleaned_merged_fairy_tales_without_eos.txt"
STORIES = {}
POEM_AUTHORS = ['EDWARD LEAR', 'ISAAC WATTS', 'JANE TAYLOR', 'PHOEBE CARY', 'ANN TAYLOR', 'ANONYMOUS', 'CHARLES KINGSLEY', 'CHARLES MACKAY', 'CLEMENT CLARKE MOORE', 'DAVID EVERETT', 'ELIZA LEE FOLLEN', 'FELICIA DOROTHEA HEMANS', 'FELICIA DOROTHEA HEMANS', 'FELICIA DOROTHEA HEMANS', 'FRANCIS C. WOODWORTH', 'FROM M. DE LAMOTTE', 'GEORGE MACDONALD', 'HANNAH FLAGG GOULD', 'HENRY WADSWORTH LONGFELLOW', 'JAMES HOGG', 'JAMES MERRICK',
                'JAMES WHITCOMB RILEY', 'JANE TAYLOR', 'JEMIMA LUKE', 'LEWIS CARROLL', 'LITTLE B. (TAYLOR?)', 'LYDIA MARIA CHILD', 'MARY HOWITT', 'MARY HOWITT', 'MARY HOWITT', 'OLD CAROL', 'REGINALD HEBER', 'RICHARD MONCKTON MILNES (LORD HOUGHTON)', 'ROBERT BURNS', 'ROBERT LOUIS STEVENSON', 'ROBERT SOUTHEY', 'SABINE BARING-GOULD', 'THOMAS HOOD', 'WILLIAM BRIGHTY RANDS', 'WILLIAM HOWITT', 'WILLIAM ROBERT SPENCER', 'WILLIAM SHAKESPEARE', 'WILLIAM WORDSWORTH']
STORY_TYPES = ['SCANDINAVIAN STORIES', 'GERMAN STORIES', 'FRENCH STORIES', 'ENGLISH STORIES','CELTIC STORIES', 'ITALIAN STORIES', 'JAPANESE STORIES', 'EAST INDIAN STORIES', 'AMERICAN INDIAN STORIES', 'ARABIAN STORIES', 'CHINESE STORIES', 'RUSSIAN STORIES', 'TALES FOR TINY TOTS', 'FANCIFUL STORIES', 'OUR CHILDREN', 'PINOCCHIO\'S ADVENTURES IN WONDERLAND[1]']

def clean_data():
    with open(data_file, "r") as f:
        lines = f.readlines()
        first_line = lines[0].strip(" \n")
        curr_title = re.sub('[\t]+', '', first_line).upper()
        for i in range(1, len(lines) - 1):
            line = lines[i].strip(" \n")
            line = re.sub('[\t]+', '', line)  # to remove tabs
            line = re.sub("        ", '', line)
            if len(line) == 0 or (line in POEM_AUTHORS) or (line in STORY_TYPES) or ("ADAPTED BY" in line):
                continue

            elif (line in ["CINDERELLA", "BLUE BEARD", "SUPPOSE!", "PRETTY COW", "THE OWL AND THE PUSSY-CAT"]):
                curr_title = line
                STORIES[curr_title] = []

            if (line == '\n' or len(line) < 3) and len(lines[i+1]) < 50:
                upcoming_title = lines[i+1].strip(" \n")
                curr_title = re.sub('[\t]+', '', upcoming_title).upper()

            elif (line[0].isnumeric()):
                curr_title = line.upper()
                STORIES[curr_title] = []

            elif (line.isupper() and ("ADAPTED BY" in lines[i+1] or "BY " in lines[i+1])):
                curr_title = line
                STORIES[curr_title] = []

            elif (line.isupper() and " STORY" in lines[i+1]):
                first_sentence = lines[i+1].split()
                if "--" in curr_title:
                    # replace with next chapter
                    curr_title = curr_title.split(
                        " --", 1)[0] + " --" + ' '.join(first_sentence[0:2])
                else:
                    curr_title = line + " --" + ' '.join(first_sentence[0:2])

                STORIES[curr_title] = [' '.join(first_sentence[2:])]

            elif (" STORY" in line or " Story." in line) and not ("OF" in line and not (" STORY" in lines[i+1])):
                first_sentence = line.split()
                if "--" in curr_title:
                    # replace with next chapter
                    curr_title = curr_title.split(
                        " --", 1)[0] + " --" + ' '.join(first_sentence[0:2])
                else:
                    curr_title = curr_title + " --" + \
                        ' '.join(first_sentence[0:2])

                begin_story = [' '.join(first_sentence[2:])]
                if len(begin_story) <= 1:
                    STORIES[curr_title] = []
                else:
                    STORIES[curr_title] = [' '.join(first_sentence[2:])]

            elif (line.isupper()) and not (str(lines[i+1].split()[0:2]).isupper() or ("THE END" in line) or ("\"" in line) or ("“" in line) or ("\'" in line) or ("{" in line) or (line in "TRESPASSERS WILL BE PROSECUTED") or (line in "FAMOUS DONKEY THE STAR OF THE DANCE") or ("ADAPTED BY" in line) or (line in "* A.D. 1482-1513")):
                if (len(line) >= 11 and len(line) < 50):
                    curr_title = line.upper()
                    STORIES[curr_title] = []

                elif (len(line) < 11 and len(line) < 50) or (line in "CHAPTER"):
                    if "--" in curr_title:
                        # replace with next chapter
                        curr_title = curr_title.split(
                            " --", 1)[0] + " --" + line
                    if not (line in curr_title):
                        curr_title = curr_title + " --" + line
                    STORIES[curr_title] = []

            elif (curr_title in STORIES) and (line.upper() != curr_title):
                STORIES[curr_title].append(line)
            else:
                STORIES[curr_title] = []

    # To finalize the cleaning: removes extra titles that never got fed in
    STORIES_COPY = STORIES.copy()
    for story in STORIES_COPY:
        if STORIES[story] == []:
            STORIES.pop(story)

    # Making each story in the dictionary a full string so that it matches the tutorial's format
    for title in STORIES:
        STORIES[title] = ' '.join(STORIES[title])


clean_data()
num_stories = len(STORIES)
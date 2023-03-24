import re
data_file = "cleaned_merged_fairy_tales_without_eos.txt"
STORIES = {}

def clean_data():
    total_stories = 0

    with open(data_file, "r") as f:
        lines = f.readlines()
        first_line = lines[0].strip(" \n")
        curr_title = re.sub('[\t.]+', '', first_line).upper()
        for i in range(1, len(lines) - 1):
            line = lines[i].strip(" \n")
            line = re.sub('[\t.].+', '', line) #to remove tabs
            if len(line) == 0:
                continue
            if (line == "\n" or len(line) < 3) and len(lines[i+1]) < 50:
                upcoming_title = lines[i+1].strip(" \n")
                curr_title = re.sub('[\t.]+', '', upcoming_title).upper()
                total_stories += 1

            if line.isupper() and not (lines[i+1].isupper() or "THE END" in line or "\"" in line or "“" in line or "\'" in line or "{" in line or "        " in line or line in "TRESPASSERS WILL BE PROSECUTED" or line in "ADAPTED BY"):
                if (len(line) > 10 and len(line) < 50):
                    curr_title = line
                    STORIES[curr_title] = []
                    total_stories += 1

                elif (len(line) <= 10 and len(line) < 50) or (line in "CHAPTER"):
                    if "--" in curr_title:
                        # replace with next chapter
                        curr_title = curr_title.split(" --", 1)[0] +  " --" + line
                    if not (line in curr_title):
                        curr_title = curr_title + " --" + line
                    STORIES[curr_title] = []
                    total_stories += 1

            if (curr_title in STORIES) and (line != curr_title):
                STORIES[curr_title].append(line)
            else:
                STORIES[curr_title] = []

def remove_unecessary_titles():
    STORIES_COPY = STORIES.copy()
    for story in STORIES_COPY:
        if STORIES[story] == []:
            STORIES.pop(story)

    print(len(STORIES_COPY))
    print(len(STORIES))

clean_data()
remove_unecessary_titles()

# print("===============================================================")
# print("===============================================================")
# print("===============================================================")
# print("===============================================================")
# print("===============================================================")
# print("===============================================================")
# print("===============================================================")
# print(STORIES)


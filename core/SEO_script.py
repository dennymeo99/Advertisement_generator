import os
import re
import shutil


def retrieve_keywords(query):
    bashCommand = "pip install seo-keyword-research-tool"
    os.system(bashCommand)

    bashCommand2 = f'seo -q "{query}" -e rs -st txt'
    os.system(bashCommand2)

    query = re.sub(" ", "_", query)
    source_path = f'{query}.txt'
    destination_path = f'../retrieved_keywords/'
    shutil.move(source_path, destination_path)
    with open(f'../retrieved_keywords/{query}.txt', 'r') as file:
        first_line = file.readline()

    os.remove(f'../retrieved_keywords/{query}.txt') #can be deleted if you want to see all the related keywords
    return first_line

if __name__ == '__main__':
    print(retrieve_keywords("Mole cola"))
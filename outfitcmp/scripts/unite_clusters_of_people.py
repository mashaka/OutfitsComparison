"""
Description: Unite clusters of photos for different people in one array
"""
import os
import pickle
from tqdm import tqdm

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CLUSTERS_DIR = os.path.join(ROOT_DIR, 'data', 'same_people_clusters')
OUTPUT_FILE = os.path.join(ROOT_DIR, 'data', 'same_people_clusters.pickle')

def unite_in_one_file():
    """ Unite clusters of photos for different people in one array """
    united_list = []
    for filename in tqdm(os.listdir(CLUSTERS_DIR)):
        with open(os.path.join(CLUSTERS_DIR, filename), encoding="utf8") as cluster_file:
            cluster = cluster_file.readlines()
            cluster = [name.strip() for name in cluster]
            united_list.append(cluster)
    with open(OUTPUT_FILE, 'wb') as output_file:
        pickle.dump(united_list, output_file)

def execute():
    """ Launch script """
    unite_in_one_file()

if __name__ == '__main__':
    execute()
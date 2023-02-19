import os
from subprocess import call

base = "."
ALL_REPOS_FILE = base + "/../dataset/final-dataset.csv"
REPO_STORE_ROOT = base + "/output"
REPOS_TO_DOWNLOAD = 200


def download_repo(repoName):
    full_repo_name = "https://github.com/" + repoName + ".git"
    folder_name = repoName.replace("/", "_")
    if not os.path.isdir(os.path.join(REPO_STORE_ROOT, folder_name)):
        os.mkdir(folder_name)
        try:
            call(["git", "clone", "--depth=1", full_repo_name, folder_name])
        except:
            print("Exception occurred!!")
            pass


file = open(ALL_REPOS_FILE, 'rt', errors='ignore')
if not os.path.isdir(REPO_STORE_ROOT):
    os.mkdir(REPO_STORE_ROOT)
os.chdir(REPO_STORE_ROOT)
for line in file.readlines()[:REPOS_TO_DOWNLOAD]:
    repo_name = line.strip('\n')
    if not repo_name == "":
        folder_name = repo_name.replace("/", "_")
        folder_path = os.path.join(REPO_STORE_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            print('downloading ' + folder_name)
            download_repo(repo_name)
print('Done.')

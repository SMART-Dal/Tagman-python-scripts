import datetime
import requests
from searchrepo import search_repo

LOC_THRESHOLD = 10000
QUALITY_SCORE_THRESHOLD = 10
QSCORED_USER_NAME = "email@test.com"
QSCORED_API_KEY = "2e300907a35b8ea1c33413876e5easdzsdea" # fake; just for example. Login to qscored.com to obtain your free key
GITHUB_API_TOKEN = "2e300907a35b8ea1c33413876e5easdzsdea" # fake; just for example. 

# Response example:
# b'[{"project_name":"BuildItBigger","uuid":"60f10a8b-35eb-4a84-a15d-f332c33e598b","score":0.63,"rank":27,"loc":143187,"lang":"java"}]'
def _get_project_info(prj_name, prj_link):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'username': QSCORED_USER_NAME,
        'Authorization': 'Token ' + QSCORED_API_KEY,
    }
    data = {
        'project_name': prj_name,
        'repo_link': prj_link,
        'lang': 'java',
        'start_index': 0,
    }
    url = "https://qscored.com/api/search_project/"
    response = requests.post(url=url, data=data, headers=headers, timeout=10 )
    if response.status_code == 200 and response.text == '{"reason":"No Project associated."}':
        return None
    print(response.content)
    return response.json()


one_year_old_date = (datetime.datetime.now().date() - datetime.timedelta(days=180))
one_year_old_date = (datetime.datetime.now().date() - datetime.timedelta(days=180))
search_repo(start_date=one_year_old_date,
            out_file='repos.csv',
            api_token=GITHUB_API_TOKEN,
            stars=40000,
            lang='Java',
            verbose=True)


with open('repos.csv', 'r') as file:
    with open('out.csv', 'w') as out_file:
        for line in file.readlines():
            tokens = line.split(' ')
            if len(tokens) > 1:
                prj_name = tokens[0].split('/')[1]
                link = 'https://github.com/' + tokens[0].strip()
                response = _get_project_info(prj_name, link)
                if response is None:
                    continue
                if response[0]['loc'] > LOC_THRESHOLD and response[0]['score'] < QUALITY_SCORE_THRESHOLD:
                    out_file.write(tokens[0].strip() + '\n')
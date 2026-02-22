from jikanpy import Jikan
from bs4 import BeautifulSoup
import requests
import time
import csv
import pandas as pd

def time_estimate(total, completed, start):
    if completed == 0:
        return None
    elapsed = time.time() - start
    iter_rate = completed/elapsed
    remaining = total - completed
    return remaining/iter_rate

jikan = Jikan()
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
manga_ds = pd.read_csv('manga_dataset.csv')
manga_ids = manga_ds['manga_id']
titles = manga_ds['title']

users = []

start = time.time()
i = 0
for id, title in zip(manga_ids, titles):
    while True:
        try:
            session = requests.Session()
            res = session.get(f"https://myanimelist.net/manga/{id}/{title.rstrip('.')}/reviews?sort=mostvoted&preliminary=off", headers=headers, timeout=10)
            res.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching content from URL: {e}")
            time.sleep(60)
            continue
        break
    soup = BeautifulSoup(res.text, 'html.parser')
    divs = soup.find_all('div', class_='review-element js-review-element')
    usercount = 0
    for div in divs:
        username = div.find('a', attrs={"data-ga-click-type": "review-manga-reviewer"}).text
        if username not in users:
            users.append(username)
            usercount += 1
        else:
            print(f"Duplicate user: {username} found, skipping...")
        if usercount >= 10:
            break
    eta = time_estimate(len(manga_ids), i, start)
    print(f"Users for manga: {title} fetched")
    if eta:
        print(f"Progress: {i}/{len((manga_ids))} - ETA: {int(eta//3600)}h {int((eta%3600)//60)}m {int(eta%60)}s remaining")
    i+=1


ratings = []

start = time.time()
i = 0
for username in users:
    while True:
        try:
            session = requests.Session()
            res = session.get(f"https://myanimelist.net/mangalist/{username}/load.json?status=1", headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            break
        except requests.exceptions.RequestException as e:
            if res.status_code == 400:
                print(f"Private mangalist, skipping...")
                break
            print(f"Error fetching mangalist from URL: {e}")
            time.sleep(60)
            continue
        break
    for item in data:
        try:
            ratings.append({'username': username, 'manga_id': item['manga_id'], 'user_score': item['score']})
        except Exception as e:
            i+=1
            continue
    eta = time_estimate(len(users), i, start)
    print(f"Mangalist of user: {username} fetched")
    if eta:
        print(f"Progress: {i}/{len((users))} - ETA: {int(eta//3600)}h {int((eta%3600)//60)}m {int(eta%60)}s remaining")
    i+=1

with open('user_dataset_extras.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=ratings[0].keys())
    writer.writeheader()
    writer.writerows(ratings)
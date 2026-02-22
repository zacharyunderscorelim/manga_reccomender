from jikanpy import Jikan
from bs4 import BeautifulSoup
import requests
import time
import csv

jikan = Jikan()
genres = jikan.genres(type="manga")['data']
for genre in genres:
    if genre['name'] == "Romance":
        url = genre['url']

manga_ds = []

for i in range(10):
    try:
        res = requests.get(f"{url}?page={i+1}", timeout=10)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {e}")

    soup = BeautifulSoup(res.text, 'html.parser')
    divs = soup.find_all('div', class_="seasonal-anime js-seasonal-anime")

    for div in divs:
        manga_link_title = div.find('a', class_="link-title")
        try:
            manga_id = int(manga_link_title.get('href').split('/')[4])
            manga_ds.append({'manga_id': manga_id, 'title': manga_link_title.text})
        except IndexError:
            print(f"Error getting id and title from {manga_link_title}")
            print(f"")
            break
    print(f"Page {i+1} title scrape complete")

iters = 0
for x in manga_ds:
    while True:
        try:
            manga_meta = jikan.manga(x['manga_id'])
            if 'data' in manga_meta:
                manga_meta = manga_meta['data']
                x['type'] = manga_meta['type']
                x['status'] = manga_meta['status']
                x['latest_update'] = manga_meta['published']['to']
                x['score'] = manga_meta['score']
                x['scored_by'] = manga_meta['scored_by']
                x['synopsis'] = manga_meta['synopsis']
                time.sleep(1)
            else:
                print('MAL timeout, skipping...')
                break
        except Exception as e:
            print(f"failed to get manga metadata with error after {iters} requests: {e}, retrying...")
            time.sleep(60)
            continue
        break
    iters+=1
    if iters%50 == 0:
        print(f"{iters} mangas metadata updated")

print(manga_ds[:5])
print(len(manga_ds))

with open('manga_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=manga_ds[0].keys())
    writer.writeheader()
    writer.writerows(manga_ds)
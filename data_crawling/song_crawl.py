from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

path = "C:/workspace/chromedriver.exe"

driver = webdriver.Chrome(path)

driver.maximize_window()
input_file = './data_crawling/korean_singer2.txt'
output_file = 'song2.csv'

singers = []
with open(input_file,'r',encoding='UTF-8') as f:
    for singer in f:
        singers.append(singer.strip())

song_db = pd.DataFrame()
titles = []
artists = []

print(singers)

for singer in singers:
    driver.get("https://www.melon.com/search/song/index.htm")
    search = driver.find_element_by_xpath('//*[@id="top_search"]')
    search.send_keys(singer)
    search.send_keys('\n')

    html = driver.page_source
    soup = BeautifulSoup(html,'html.parser')

    try:
        view_list = soup.find("tbody").find_all('tr')
    except:
        print('no result')

    for l in view_list:
        try:
            title = l.find('a','fc_gray').get_text()
            titles.append(title)
            artist = l.find('a','fc_mgray').get_text()
            artists.append(artist)

        except:
            pass

song_db['titles'] = pd.Series(titles)
song_db['artist'] = pd.Series(artists)

    
song_db.to_csv(output_file)
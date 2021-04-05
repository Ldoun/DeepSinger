from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

path = "C:/workspace/chromedriver.exe"

driver = webdriver.Chrome(path)

driver.maximize_window()

input_file = 'data_list.csv'
output_file = 'song_lyrics.csv'

song_data = pd.read_csv('data_crawling/data_list.csv')

'''for i in range(len(song_data)):
    title = song_data[['titles','artist']].iloc[i]['titles']
    artist = song_data[['titles','artist']].iloc[i]['artist']
    
    driver.get("https://www.melon.com/search/song/index.htm")
    search = driver.find_element_by_xpath('//*[@id="top_search"]')
    search.send_keys(title + artist)
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

    
song_db.to_csv(output_file)'''

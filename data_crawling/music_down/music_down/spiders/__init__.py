# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.

#https://www.youtube.com/results?search_query=%EC%9A%B4%EB%AA%85%EC%9D%B4+%EB%82%B4%EA%B2%8C+%EB%A7%90%ED%95%B4%EC%9A%94+%ED%97%A4%EC%9D%B4%EC%A6%88+%28Heize%29+%EA%B0%80

import scrapy
import pandas as pd
import json
import re
from pytube import YouTube
import os
import pandas as pd
import time

class QuotesSpider(scrapy.Spider):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.check_progress = '/content/drive/MyDrive/data/svs/progress.csv'
        self.output_path = '/content/drive/MyDrive/data/svs'
        self.input_list = '/content/drive/MyDrive/data/svs/lyrics_result_drop.csv'

        if os.path.isfile(self.check_progress):
            self.progress_db = pd.read_csv(self.check_progress)
        else:
            self.progress_db = pd.DataFrame(columns = ['titles' , 'artist','video_name'])
        
        self.video_series = list(self.progress_db['video_name'])
        self.title_series = list(self.progress_db['titles'])
        self.artist_series = list(self.progress_db['artist'])

        self.result_db = pd.read_csv(self.input_list)

    name = "quotes"
    def make_url(self):
        data = self.result_db[['titles','artist']]
        if len(self.progress_db) != 0:
            data = pd.concat([data,self.progress_db[['titles','artist']]]).drop_duplicates(keep=False)
            print('here')
            print(data.head())
            if len(data) == 0:
                self.result_db['video_name'] = pd.Series(self.video_series)
                self.result_db.to_csv('f_result.csv')
                print('finished')
        

        data['adding'] = data['titles'] + ' ' + data['artist'] + ' 가사'
        
        urls = []
        info = []
        for i,j,k in zip(data['adding'],data['titles'],data['artist']):
            urls.append('https://www.youtube.com/results?search_query='+i)
            info.append([['title',j],['artist',k]])
        return urls,info

    def start_requests(self):
        urls, info = self.make_url()
        for url,meta in zip(urls,info):
            yield scrapy.Request(url=url, callback=self.parse,meta=meta)

    def parse(self, response):
        pattern = r'\bvar\s+data\s*=\s*(\{.*?\})\s*;\s*\n'
        links = response.css('script::text')
        for i in links:
            time.sleep(0.2)
            try:
                print(response.meta)
                i_data = i.get()  
                if 'var ytInitialData' in i.get():
                    link = re.search('watch\?v=[^/]{11}',i_data.replace('var ytInitialData =','')).group()

                    url = 'https://www.youtube.com/' + link 
                    out_file = YouTube(url).streams.filter(only_audio = True).first().download(output_path = '/content/drive/MyDrive/data/svs/music')
                    base, ext = os.path.splitext(out_file)
                    new_file = base + '.mp3'
                    os.rename(out_file, new_file)
                    print("music has been successfully downloaded.")

                    self.title_series.append(response.meta['title'])
                    self.artist_series.append(response.meta['artist'])
                    self.video_series.append(new_file)

                    self.progress_db = pd.DataFrame()
                    self.progress_db['titles'] = pd.Series(self.title_series)
                    self.progress_db['artist'] = pd.Series(self.artist_series)
                    self.progress_db['video_name'] = pd.Series(self.video_series)
                    
                    self.progress_db.to_csv(self.check_progress)
                
                else:
                    continue
            except Exception as e:
                print(e)   
        print(self.progress_db.tail())
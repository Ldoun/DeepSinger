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

class QuotesSpider(scrapy.Spider):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.save_result_path = 'result.csv'
        self.output_path = '/content/drive/MyDrive/data/svs'
        self.input_list = '../data_list.csv'

        if os.path.isfile(save_result_path):
            self.song_db = pd.read_csv(save_result_path)
        else:
            self.song_db = pd.DataFrame(columns = ['title' , 'artist', 'video_name'])

    name = "quotes"
    def make_url(self):
        data = pd.read_csv(self.input_list)
        data = data[['titles','artist']]
        if len(self.song_db) != 0:
            data = data[data['titles'] == self.song_db['title'] and data['artist'] == self.song_db['artist']]
        
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
            try:
                print(response.meta)
                i_data = i.get()  
                if 'var ytInitialData' in i.get():
                    link = re.search('watch\?v=[^/]{11}',i_data.replace('var ytInitialData =','')).group()

                    url = 'https://www.youtube.com/' + link 
                    out_file = YouTube(url).streams.filter(only_audio = True).first().download(output_path = '/content/drive/MyDrive/data/svs')
                    base, ext = os.path.splitext(out_file)
                    new_file = base + '.mp3'
                    os.rename(out_file, new_file)
                    print("music has been successfully downloaded.")
    
                    self.song_db = self.song_db.append({'title':response.meta['title'],'artist':response.meta['artist'],'video_name':out_file},ignore_index=True)
                    self.song_db.to_csv(self.save_result_path)
                
                else:
                    continue
            except Exception as e:
                print(e)   
        print(self.song_db.head()) 
# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
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
        custom_settings = {
        'DOWNLOADER_MIDDLEWARES' : {
                'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
                'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
                'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
                'scrapy_fake_useragent.middleware.RetryUserAgentMiddleware': 401,
            }
        }

        self.save_result_path = '/content/drive/MyDrive/data/svs/melone_result_re.tsv'
        self.input_list = '../data.tsv'

        if os.path.isfile(self.save_result_path):
            self.song_db = pd.read_csv(self.save_result_path,sep='\t')
        else:   
            self.song_db = pd.DataFrame(columns = ['titles' , 'artist', 'lyrics'])
        
        self.lyrics_series = list(self.song_db['lyrics'])
        self.title_series = list(self.song_db['titles'])
        self.artist_series = list(self.song_db['artist'])
    name = "quotes"

    def make_url(self):
        data = pd.read_csv(self.input_list,sep='\t')
        data = data[['titles','artist']]
        if len(self.song_db) != 0:
            data = pd.concat([data,self.song_db[['titles','artist']]]).drop_duplicates(keep=False)
            print(data.head())
            print('????????????????')
            if len(data) == 0:
                print('finished')
        data['adding'] = data['titles'] + ' ' + data['artist']
        
        urls = []
        info = []
        for i,j,k in zip(data['adding'],data['titles'],data['artist']):
            urls.append('https://www.melon.com/search/total/index.htm?q='+i)
            #urls.append('https://www.melon.com')
            info.append([['title',j],['artist',k]])
        return urls,info

    def start_requests(self):
        urls, info = self.make_url()
        for url,meta in zip(urls,info):
            yield scrapy.Request(url=url, callback=self.parse,meta=meta)

    def parse(self, response):
        song_id = response.xpath('//*[@id="conts"]/div[*]/div/ul/li[1]/dl/dt/a[1]/@data-song-no').getall()[0]
        #print('song_id: '+str(song_id.get()))
        url = 'https://www.melon.com/song/detail.htm?songId='+str(song_id)
        yield scrapy.Request(url=url, callback=self.parse_lyrics,meta=response.meta)

    def parse_lyrics(self,response):
        data = response.xpath('//*[@id="d_video_summary"]/text()').getall()
        lyrics = re.sub('\\r\\n(\\t){1,}','', '%'.join(data))
        if len(data) == 1:
            lyrics = re.sub('(\\r\\n){1,}','%',lyrics) 
        
        #raw_data = {'titles':response.meta['title'],'artist':response.meta['artist'],'lyrics':lyrics.replace("\n","")}
        self.title_series.append(response.meta['title'])
        self.artist_series.append(response.meta['artist'])
        self.lyrics_series.append(lyrics)
        self.song_db = pd.DataFrame()
        self.song_db['titles'] = pd.Series(self.title_series)
        self.song_db['artist'] = pd.Series(self.artist_series)
        self.song_db['lyrics'] = pd.Series(self.lyrics_series)
        self.song_db.to_csv(self.save_result_path,sep='\t',index=False)
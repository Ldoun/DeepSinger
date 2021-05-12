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

        self.save_result_path = '/content/drive/MyDrive/data/svs/with_seperation_mark_result.csv'
        self.input_list = '../data/for_crawl_seperation_mark_dat_list.csv'

        if os.path.isfile(self.save_result_path):
            self.song_db = pd.read_csv(self.save_result_path)
        else:   
            self.song_db = pd.DataFrame(columns = ['titles' , 'artist', 'lyrics'])

        self.genie_lyrics_series = list(self.song_db['genie_lyrics'])        
        self.melone_lyrics_series = list(self.song_db['melone_lyrics'])
        self.title_series = list(self.song_db['titles'])
        self.artist_series = list(self.song_db['artist'])

    name = "quotes"
    def make_url(self):
        data = pd.read_csv(self.input_list,sep='\t')
        #data = data[['titles','artist']]
        if len(self.song_db) != 0:
            data = pd.concat([data,self.song_db[['titles','artist']]]).drop_duplicates(keep=False)
            print(data.head())
            print('????????????????')
            print(len(data))
            if len(data) == 0:
                print('finished')

        data['adding'] = data['titles'] + ' ' + data['artist']
        
        
        urls = []
        info = []
        for i,j,k in zip(data['adding'],data['titles'],data['artist']):
            urls.append('https://www.genie.co.kr/search/searchMain?query='+i)
            info.append([['title',j],['artist',k]])

        return urls,info

    def start_requests(self):
        urls, info = self.make_url()
        for url,meta in zip(urls,info):
            yield scrapy.Request(url=url, callback=self.parse_genie,meta=meta)

    def parse_genie(self, response):
        #song_id = response.xpath('//*[@id="frm_songList"]/div/table/tbody/tr/td[1]/div/input/@value') melone
        song_id = response.xpath('//*[@id="body-content"]/div[3]/div[2]/div/table/tbody/tr[1]/@songid')
        #print('song_id: '+str(song_id.get()))
        url = 'https://www.genie.co.kr/detail/songInfo?xgnm='+str(song_id.get())
        yield scrapy.Request(url=url, callback=self.parse_genie_lyrics,meta=response.meta)

    def parse_genie_lyrics(self,response):
        # response.xpath('//*[@id="d_video_summary"]/text()') melone
        #lyrics = re.sub('\\r\\n(\\t){1,}','', ' '.join(response.xpath('//*[@id="pLyrics"]/p/text()').getall()))
        #raw_data = {'titles':response.meta['title'],'artist':response.meta['artist'],'lyrics':lyrics.replace("\n","")}
        
        lyrics = re.sub('(\r\n){1,}','%', ' '.join(response.xpath('//*[@id="pLyrics"]/p/text()') .getall()))
        lyrics = lyrics.replace('\n','%')
        lyrics = re.sub('.*작사.*작곡.*%%','', lyrics)

        info = []
        info.append([['title',response.meta['title']],['artist',response.meta['artist']],['genie',lyrics]])
        
        url = "https://www.melon.com/search/song/index.htm?q=" + response.meta['title'] +' '+ response.meta['artist']
        yield scrapy.Request(url=url, callback=self.parse_melone,meta=info)
    
    def parse_melone(self,response):
        song_id = response.xpath('//*[@id="frm_songList"]/div/table/tbody/tr/td[1]/div/input/@value')
        #print('song_id: '+str(song_id.get()))
        url = 'https://www.melon.com/song/detail.htm?songId='+str(song_id.get())
        yield scrapy.Request(url=url, callback=self.parse_melone_lyrics,meta=response.meta)
        
    def parse_melone_lyrics(self,response):

        lyrics = re.sub('\\r\\n(\\t){1,}','', ' '.join(response.xpath('//*[@id="d_video_summary"]/text()').getall()))
        lyrics = lyrics.replace('\n','%')

        self.title_series.append(response.meta['title'])
        self.artist_series.append(response.meta['artist'])
        self.genie_lyrics_series.append(response.meta['genie'])
        self.melone_lyrics_series.append(lyrics)

        self.song_db = pd.DataFrame()
        self.song_db['titles'] = pd.Series(self.title_series)
        self.song_db['artist'] = pd.Series(self.artist_series)
        self.song_db['genie_lyrics'] = pd.Series(self.genie_lyrics_series)
        self.song_db['melone_lyrics'] = pd.Series(self.melone_lyrics_series)

        self.song_db.to_csv(self.save_result_path)
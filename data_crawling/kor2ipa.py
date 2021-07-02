from phonemizer import phonemize
import sys

sentence = sys.stdin.readlines()    
num_cnt = len(sentence)
with open('data/lyric_ipa.re_sub.new_data.tsv','w') as f:
    for words in sentence:
        word = words.split(' ')
        for w in word:
            ipa = phonemize(w,language='ko',backend='espeak',language_switch='remove-flags',preserve_punctuation=True,punctuation_marks='%')
            '''if '(' in w and ')' in w:
                f.write('('+ipa+')')
            elif '(' in w:
                f.write('(')
                f.write(ipa )
            elif ')' in w:
                f.write(ipa + ')')
            else:
                f.write(ipa )'''
            f.write(ipa)
        f.write('\n')
        print(num_cnt)
        num_cnt -= 1 
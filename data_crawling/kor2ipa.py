from phonemizer import phonemize
import sys

sentence = sys.stdin.readlines()    
num_cnt = len(sentence)
with open('data/seperation_mark_2ipa.txt','w') as f:
    for words in sentence:
        word = words.split(' ')
        for w in word:
            ipa = phonemize(w,language='ko',backend='espeak',language_switch='remove-flags',preserve_punctuation=True,punctuation_marks='%()')
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

'''phonemize(
        text,
        language='en-us',
        backend='festival',
        separator=default_separator,
        strip=False,
        preserve_punctuation=False,
        punctuation_marks=Punctuation.default_marks(),
        with_stress=False,
        language_switch='keep-flags',
        njobs=1,
        logger=get_logger()):'''
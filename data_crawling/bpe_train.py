import sentencepiece as spm
import sys

file = sys.argv[1]
model_name = sys.argv[2]
vocab_size = int(sys.argv[3])

spm.SentencePieceTrainer.train(
    f"--input={file} --model_prefix={model_name} --vocab_size={vocab_size + 4}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + 
    " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
    "--character_coverage=1.0" 
) 
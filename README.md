End of development indefinitely
- [X] song lyric recognition
- [ ] voice synthesis model

### What

---

- My Hish School Graduation project and my very first academic paper implementation project(without any code source)
- I was only able to develop the first model(lyric alignment model) only. The data I collected wasn’t clear enough for the next step model
- project's purpose was to synthesize Korean songs given users' voices.
- developed based on the **DeepSinger: Singing Voice Synthesis with Data Mined From the Web**

### How

---

- By studying the previous research, I develop everything from the scratch from data collection and model training
- data collection and preprocessing
    - using scrapy with python I collected audio from youtube and lyrics from Melone.
    - using Spleeter I removed the MR from the audio. (The author used the data from the recording room, but I wasn’t able to get my hands on such data)
    - normalization of audio volume.
    - using some algorithms, remove parts that have no vocals.
    - turn lyrics to IPA(international phonetic alphabet)
    - filter some song that doesn’t qualify(such as long songs, multi vocals)
- The paper is composed of two models. one for finding out what audio frames match the lyrics and the other for the voice synthesis.
- lyric alignment model
    ![1234](https://github.com/Ldoun/DeepSinger/assets/67096173/4052e5c3-c422-49da-ba90-569263bf674e)
    
    - Seq2Seq
        - Seq2Seq model with Encoder, Decoder
        - Input: Song →  recognize lyrics from it
        - using the model’s attention we can find out the relation between frames and IPA
        - It is important to train a model with a clear attention graph. (look at the below images)
    - location sensitive guided attention
        - given previous attention values and the output of the encoder and decoder we calculate the attention
        - as the attention differs from a diagonal line we give the model a bigger loss
    - variation of truncated back-propagation through time(training algorithm)
        - when we train the model, the model has to exceed a score of 0.6 in attention for it to step over the next parts of the song.

### Attention Plotting(to see how well our model is training)

---
![123](https://github.com/Ldoun/DeepSinger/assets/67096173/2c97f03c-1397-43c5-abbc-4d427ad426bd)

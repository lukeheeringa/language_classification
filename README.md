#### Capstone: Language Classification
Luke Heeringa

Data Science Immersive Remote (DSIR-113020)

March 4, 2021

#### Problem Statement

Can machine learning be used to identify the language being spoken in an audio clip? 

#### Background Summary

This project uses neural networks to classify audio clips as belonging to one of five major world languages. The ability to identify spoken language could have several applications, such as being used to connect clients with appropriate interpretation services over the telephone or as a tool to contextualize what language is being spoken before basic AI attempts to transcribe or translate audio or video recordings. 

The five languages chosen as classes are English (en), Spanish (es), French (fr), Russian (ru), and Mandarin Chinese (zh). File names and code generally refer to them by their ISO 639-1 two letter abbreviations indicated in parentheses. These languages are five of the eight most spoken languages worldwide, each with over 250 million fluent speakers (the other three are Hindi, Arabic, and Bengali, but those samples are much less common online). 

#### Data Summary

Data samples consisted of 5 second clips of human speech sampled at a frequency of 16,000 kHz. Samples came from five different online sources with a variety of speaker genders, ages, dialects, background noise, and audio quality. Any clips in the training set longer than 5 seconds may have been split into multiple individual clips for additional information, but the validation set remained uncontaminated. 

The python librosa library was used to load audio files in .wav or .mp3 formats into time series of 80,000 points (5 seconds * 16kHz). These timeseries were then transformed into their Mel frequency cepstrum coefficients (mfccs), a technique using Fourier transformation and a rolling window time period to reduce a sound wave into components. Each additional coefficient calculated represents a diminishing portion of the overall variance, so 10 mfccs were chosen to balance computational complexity while preserving audio information. 

#### Model Summary

The final data were then matrices of 10 rows and 157 columns, representing 10 layered time series across a span of 157 rolling average frequencies in the clip. Over 80,000 samples were used for training, with over 8,000 reserved for validation. The classes were approximately evenly distributed in both the training and validation sets, with each language representing between 18 and 21% of the whole. The null accuracy of predicting every sample to be English was 19.8%. 

The matrices were used as inputs for several tested neural network architectures. Dense layer only models reached a maximum final validation accuracy of 37.5%, or not quite twice the null accuracy. The most effective architecture was found to be the use of gated recurrent units (GRUs). The maximum final validation accuracy of the GRU model with bidirectionality was 58.2%, or just under three times the null accuracy. The large separation between the validation and training accuracies (58.2 vs 77.0) indicates that there is still potential to improve the model either with stricter sample quality verification, alternative processing techniques, or possibly the implementation of a pre-trained network with audio represented as spectrograms. 

The final model most accurately classified the Russian language, correctly identifying 79.5% of the Russian validation clips. The worst performing language was English, with only 40.1% of English validation clips being correctly identified. One potential cause of the poor performance on the English language may be that the samples represented the largest variety of speakers, as although all five languages have numerous dialects, English, as the most commonly spoken language in the world, likely has the most internet contributers of varying national identity and first language. The most common misclassifications were all between the three western European languages of English, Spanish, and French, possibly showing that Russian and Mandarin were unique enough to be identified more often.  

#### Application

The streamlit application included in the repository will be launched in the near future. 

#### Audio Data Sources

- [Audio Lingua](https://www.audio-lingua.eu/?lang=en)
- [Common Voice](https://commonvoice.mozilla.org/en/datasets)
- [EveryTongue](http://www.everytongue.com/)
- [Omniglot](https://omniglot.com/soundfiles/)
- [VoxForge](http://www.voxforge.org/home/downloads)


#### File Directory
- README.md

- presentation_slides.pdf


- code 
    - 01_audiolingua_scraping.ipynb : data collection, part 1
    - 02_everytongue_scraping.ipynb : data collection, part 2
    - 03_voxforge_scraping.ipynb : data collection, part 3
    - 04_audio_processing.ipynb : data processing, part 1
    - 05_mfcc_processing.ipynb : data processing, part 2
    - 06_cv_processing.ipynb : data processing, part 3
    - 07_age_and_gender.ipynb : exploratory data analysis, part 1
    - 08_mfcc_EDA.ipynb : exploratory data analysis, part 2
    - 09_dense_model.ipynb : modeling, part 1
    - 10_gru_model.ipynb : modeling, part 2


- models
    - gru_model_split552.h5 : the final keras model built in notebook 10


- streamlit
    - classification_app.py : streamlit application with audio recording and prediction capability


- Procfile : used for launching streamlit application on heroku

- setup.sh : used for launching streamlit application on heroku

- requirements.txt : a listing of the virtual environment requirements used for the streamlit app ()

- full_requirements.txt : a complete listing of the virtual environment requirements used for the entire project

- .gitignore : a listing of the files to be excluded from the git repository due to size or lack of necessity

Additionally, the following directories were used to build this project, but were excluded due to file size: 

- audio
    - 1_audiolingua (subfolders 'en' and 'zh') : audio files from audio-lingua
    - 2_everytongue (subfolders 'es', 'fr', 'ru', and 'zh') : audio files from everytongue
    - 3_omniglot (subfolders 'en', 'es', 'fr', 'ru', and 'zh') : audio files from omniglot
    - 4_voxforge (subfolders 'en', 'es', 'fr', and 'ru') : audio files from voxforge
    - v_commonvoice (subfolders 'en', 'es', 'fr', 'ru', and 'zh') : audio files from commonvoice
 
 
 - data
    - timeseries (subfolders 'en', 'es', 'fr', 'ru', and 'zh') : saved timeseries representations of audio clips
    - training  : saved mfcc and target arrays for sources 1-4
    - validation : saved mfcc and target arrays for commonvoice clips

#### Special Thanks

Brandon Bergeron, Chuck Dye, Jeff Hale, Claire Hester, Jacob Koehler, & Adam Pardo

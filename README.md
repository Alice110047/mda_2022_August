# mda_2022_August
This is the submission file for retake exam in August. Here is the brief introduction for each file.
## udf.py
This file contains the user defined functions:
* get_url(): extract the links of each speech from the website https://www.americanrhetoric.com/barackobamaspeeches.htm.  
* get_response(): obtain the titles and the texts of speeches.
* extract_date(): extract the date of indivual speech.
* allowed(): keep the useful strings.
* title(): remove strings like "[<title>")" and "</title>".
* remove(): remove the words labeled by "PERSON", "ORG" and "DATE" after named entity recognition using spaCy.
* most_occur(): count 100 most frequent words.
* blob_sentiment(): calculate the polarity.
* getAnalysis(): determine the positve, neutral and negative polarity.
## request_data.ipynb
This notebook is used to download data from the web https://www.americanrhetoric.com/barackobamaspeeches.htm. A dataframe is created and saved as "obama_speech.csv"
## data_preprocessing.ipynb
Use spaCy to preprocess the data for sentiment analysis and topic modeling. Save the preprocessed data as "obama_speech_preprocessed.csv"
## sentiment.ipynb
Use TextBlob to do a sentiment analysis
## 

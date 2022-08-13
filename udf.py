from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.corpus import stopwords
from collections import Counter

StopWords = stopwords.words("english")
StopWords.extend(
    ["u", "from", 'obama', 'barack', 'america', 'transcription', 'bankmovie', 'rhetoricbarackobama', 'andas', 'today',
     'everyday', 'everybody', 'speechesonline', 'copyright', 'audit', 'debatethird','video)barackobamaremark','transcript',
     'mp3','pdf','CD','Book'])
import spacy
from textblob import TextBlob


class scrab:
    def __init__(self):
        return

    def get_url(url):
        thepage = urlopen(url)
        soup = BeautifulSoup(thepage, "lxml")
        web_list = []
        for a_href in soup.find_all("a", href=True):
            web_list.append(a_href["href"])
        speech = []
        for i in web_list:
            if i.startswith("speeches"):
                speech.append(i)
        speech_web = []
        for i in speech:
            if i.endswith('htm'):
                speech_web.append(i)
        start = speech_web.index('speeches/barackobama/barackobamainauguraladdress.htm')
        speech_president = speech_web[start:]
        # remove replicate
        new = []
        for i in speech_president:
            if i not in new:
                new.append(i)
        speech_url = []
        for val in range(len(new)):
            speech_url.append("https://www.americanrhetoric.com/" + str(new[val]))
        return speech_url

    def get_response(x):
        resp = urlopen(x)
        s = BeautifulSoup(resp.read(), "lxml")
        row = dict()
        row['title'] = s.find_all("title")
        row['text'] = s.get_text(strip=True)
        return row

    def extract_date(page):
        r = urlopen(page)
        s = BeautifulSoup(r.read(), "lxml")
        date = []
        result = s.find_all("b")
        for a in result:
            if a:
                if a.text and len(a.text) > 3:
                    date.append(a.text)
        return date


class clean:
    def __init__(self):
        return

    def allowed(speech):
        allowed = speech[speech.find("transcribed directly from audio]")
                         + len("transcribed directly from audio]"):speech.find(
            "Book/CDs by Michael E. Eidenmuller")].strip()
        return allowed

    def title(x):
        x = str(x)
        x = x[x.find("[<title>") + len("American"):x.find("</title>")]
        return x

    def remove(text):
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        new = text
        for e in reversed(doc.ents):
            if e.label_ == "PERSON" or e.label_ == "ORG" or e.label_ == 'DATE':
                new = new[:e.start_char] + new[e.start_char + len(e.text):]
        return new

    def most_occur(self):
        all_txt = self.to_list()
        all_txt_str = ' '.join(all_txt)
        all_txt_list = all_txt_str.split()
        Count = Counter(all_txt_list)
        most_occur = Count.most_common(100)
        return most_occur


class sentiment:
    def __init__(self):
        return

    def blob_sentiment(txt):
        sent = TextBlob(txt).sentiment.polarity
        return sent

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

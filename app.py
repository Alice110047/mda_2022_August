import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import udf
import spacy
from nltk import word_tokenize
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

app = dash.Dash(__name__, title='MDA 2022 August')
server = app.server

##  Sentiment analysis
df = pd.read_csv('obama_speech_preprocessed.csv')
df_sentiment = df.drop(columns=['processed_for_lda'])
df_sentiment['sentiment'] = df_sentiment['processed_for_sentiment'].apply(lambda x: udf.sentiment.blob_sentiment(x))
df_sentiment['analysis'] = df_sentiment['sentiment'].apply(lambda x: udf.sentiment.getAnalysis(x))
df_sentiment['sentiment_comsum'] = df_sentiment['sentiment'].cumsum()

fig = make_subplots(rows=2, cols=2, subplot_titles=("Sentiment Score", "Cumulative Score", "Positive or Negative"
                                                    ,'Histogram of Score'))
fig.add_trace(go.Scatter(x=df_sentiment.index, y=df_sentiment['sentiment'],name='polarity'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df_sentiment.index, y=df_sentiment['sentiment_comsum'],name='cumulative polarity'), row=1, col=2)
fig.add_trace(go.Histogram(x=df_sentiment['analysis'],name='positive or negative'), row=2, col=1)
fig.add_trace(go.Histogram(x=df_sentiment['sentiment'],xbins=dict(
        start=-1.0,
        end=0.5,
        size=0.05),name='distribution of polarity'), row=2, col=2)
fig.update_layout(height=800, width=1800)

## Topic modelling LDA
df_lda = df.drop(columns=['processed_for_sentiment'])
en = spacy.load('en_core_web_lg')
stopwords = en.Defaults.stop_words
Stop_Words = list(stopwords)
Stop_Words.extend(
    ["u", "from", 'obama', 'barack', 'america', 'transcription', 'bankmovie', 'rhetoricbarackobama', 'andas', 'today',
     'everyday', 'everybody', 'speechesonline', 'copyright', 'audit', 'debatethird', 'video)barackobamaremark',
     'transcript','mp3', 'pdf', 'CD', 'Book', 'american', 'president', 'country','year', 'day', 'thank', 'way', 'thing',
     'get', 'something', 'everything', 'nation', 'world', 'let', 'state','idea', 'try','part', 'place', 'time', 'fact',
     'talk', 'look', 'kind', 'problem', 'end', 'progress', 'program', 'use', 'example', 'face', 'point', 'challenge',
     'history', 'meet', 'make', 'month', 'decade', 'member', 'start', 'step', 'see', 'cut', 'office', 'use', 'point',
     'term', 'number', 'question','government', 'greet', 'meantime', 'god', 'americans', 'mr.', 'evening', 'afternoon',
     'morning'])

# Remove stopword
def stop(x):
    t = word_tokenize(x)
    tok = [word for word in t if not word in Stop_Words]
    d = " ".join(tok)
    return d


df_lda['doc_lda'] = df_lda['processed_for_lda'].apply(lambda x: stop(x))


def freq_filter(x):
    words: list[str] = nltk.word_tokenize(x)
    fd = nltk.FreqDist(words)
    cc = []
    for i in words:
        if fd.freq(i) < 5:
            cc.append(i)
    fil = " ".join(cc)
    return fil


df_lda['doc_filter'] = df_lda['doc_lda'].apply(lambda x: freq_filter(x))

## Training a LDA model
corpus = df_lda['doc_filter']
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english')
tf_matrix = vectorizer.fit_transform(corpus)
tf_feature_names = vectorizer.get_feature_names_out()

## Perplexity
plexs = []
number_topic = list(range(2, 20))
for i in number_topic:
    lda = LatentDirichletAllocation(n_components=i, max_iter=20,
                                    learning_method='online',
                                    random_state=0)
    lda.fit(tf_matrix)
    plexs.append(lda.perplexity(tf_matrix))

## Perplexity
fig2 = px.line(x=number_topic, y=plexs, title="perplexity figure")
fig2.update_layout(height=600, width=1800)

## Time series plots
## Determine the number of topic corresponding to the lowest value of perplexity
plex = pd.DataFrame(list(zip(number_topic,plexs)), columns=['topic number','perplexity'])
plex.sort_values("perplexity",inplace=True)
num_topic = plex.iloc[0,0]
lda = LatentDirichletAllocation(learning_method='online', n_components=num_topic, random_state=0, learning_decay=0.5)
lda.fit(tf_matrix)
num_top_words = 20
lda_components = lda.components_
topic_top_words = []
for index, component in enumerate(lda_components):
    zipped = zip(tf_feature_names, component)
    top_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:num_top_words]
    top_terms_list = list(dict(top_key).keys())
    topic_top_words.append(top_terms_list)

# Load topic distribution
doc_topic_dist = lda.transform(tf_matrix).tolist()
df_lda['topic_distributions'] = pd.Series(doc_topic_dist)
topic_distributions_df = df_lda['topic_distributions'].apply(pd.Series)
topic_distributions_df.columns = [" ".join(topic[:5]) for topic in topic_top_words]
#Merge that column into the dataframe
df_lda_for_plot = pd.concat([df_lda, topic_distributions_df], axis=1)
# Convert to datetime
df_lda_for_plot['date'] = pd.to_datetime(df_lda_for_plot['date'])
# Extract year
df_lda_for_plot['year'] = pd.to_datetime(df_lda_for_plot['date'].dt.year, format='%Y')
# Extract year and month
df_lda_for_plot['year-month'] = df_lda_for_plot['date'].dt.to_period('M')
df_lda_for_plot['Date (by month)'] = [month.to_timestamp() for month in df_lda_for_plot['year-month']]
# Set year and month as Pandas Index
df_lda_for_plot = df_lda_for_plot.set_index('Date (by month)')
df_lda_for_plot_by_mean = df_lda_for_plot.groupby(df_lda_for_plot.index).mean()
topic_num_range = list(range(num_topic))
topic_label = []
for i in topic_num_range:
     topic_number = i
     topic_label_ind = ' '.join(topic_top_words[i][:5])
     topic_label.append(topic_label_ind)

fig3 = make_subplots(rows=int(num_topic), cols=1)
for i in topic_num_range:
       fig3.add_trace(go.Scatter(x=df_lda_for_plot_by_mean.index, y=df_lda_for_plot_by_mean.iloc[:,i],
                    mode='lines',
                    name=f'Speeches By Topic: \n{topic_label[i]}'),
                     row=i+1, col=1)
fig3.update_layout(height=1200, width=1800)

app.layout = html.Div(children=[
    html.H1(children='Obama Speeches'),

    html.Div(children='''
        A sentiment analysis using Textblob.
    '''),

    dcc.Graph(
        id='sentiment-graph',
        figure=fig
    ),

    html.Div(children='''Perplexity'''),
    dcc.Graph(
        id='perplexity-graph',
        figure=fig2
    ),

    html.Div(children='''Topic Time Series Plot'''),
    dcc.Graph(id='topic-graph',
              figure=fig3)
])

if __name__ == '__main__':
    app.run_server(debug=True)


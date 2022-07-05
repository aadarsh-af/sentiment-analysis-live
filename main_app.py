from flask import Flask, render_template, request, session
import pandas as pd
from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
# sentiment_analysis.save_pretrained("sentiment_analysis_model/")
# sentiment_analysis = pipeline('sentiment-analysis', model="sentiment_analysis_model/")


def classifier(text):
    try:
        return sentiment_analysis(text)[0]['label']
    except RuntimeError:
        print(f"Error occurred for {text}")


def huggingface_sentiment_scores(data):
    data['Text'] = data['Text'].astype("str")
    data['Sentiment'] = data['Text'].apply(lambda x: classifier(x))
    positive = data[(data['Star'] < 3) & (data['Sentiment'] == "POSITIVE")]
    negative = data[(data['Star'] >= 3) & (data['Sentiment'] == "NEGATIVE")]
    pn = positive.append(negative, ignore_index=True)

    return pn


# WSGI Application
app = Flask(__name__)

app.secret_key = 'You Will Never Guess'


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file)
        session['uploaded_csv_file'] = df.to_json()
        return render_template('uploaded.html')


@app.route('/show_data')
def showData():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Convert dataframe to html format
    uploaded_df_html = uploaded_df.to_html()
    return render_template('display.html', data=uploaded_df_html)


@app.route('/sentiment')
def SentimentAnalysis():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = huggingface_sentiment_scores(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    return render_template('display.html', data=uploaded_df_html)


if __name__ == '__main__':
    app.run(debug=True, port=5003)

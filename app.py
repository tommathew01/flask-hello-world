from flask import Flask, jsonify, request
import requests
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


API_URL = "https://api-inference.huggingface.co/models/google/pegasus-cnn_dailymail"
headers = {"Authorization": "Bearer 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"}
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello world"



@app.route('/flutterpart', methods=['POST'])
def my_endpoint():
    text = request.form['textforsummarization']
    # text2 = request.json
    # print(text2)
    return text



def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


@app.route('/extsumm')
def sq():
    stopwords= list(STOP_WORDS)
    text = """A. So where will you study? I heard you are going abroad next year.
    B.I’m planning to go to Canada. I applied to a few universities but I’m still waiting for confirmation. I haven’t been accepted yet.
    A.Oh that’s great. Are you excited? What will you study?
    B.Very excited. You remember I told you I wanted to study journalism?
    A.Yes, of course.
    B.Well. Some of the graduate programs in Canada are very good… and they offer internships. I can work for a media company and maybe get a job there in the future.
    A.Well.. I hope it works out. How about the weather? Isn’t it cold. I had a friend that went to Canada and all he talked about was the cold weather. Negative 50 degrees.
    B. Yes, it’s very cold. But I like snow. It can’t be that bad.
    A. NEGATIVE 50. Are you crazy!
    B. It’s very international. If other people can do it, I can do it.
    A. Hopefully. Well.. it looks beautiful and I heard the food is really good. They have everything there.
    B. Yeah. We can eat a different type of food every day. Pho on Mondays. Sushi on Tuesdays. Italian on Wednesdays. Just thinking about it is making me hungry.
    A. Good luck! Let me know when you find out."""

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    # print(tokens)
    punctuationToAvoid = punctuation +'\n'
    # punctuationToAvoid
    type(punctuationToAvoid)
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuationToAvoid:
                if word.text.lower() not in word_frequencies.keys():
                    word_frequencies[word.text.lower()] = 1
                else:
                    word_frequencies[word.text.lower()] += 1
    # word_frequencies
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] /= max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    # sentence_tokens
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    # sentence_scores
    from heapq import nlargest
    select_length = int(len(sentence_tokens)*0.6)
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    summary = [word.text for word in summary]
    finalsummary = ''.join(text)
    output = query({
    "inputs": finalsummary,
    })
    print(output)
    # output = output.replace("<n>", " ")
    # output = list(map(lambda x: x.replace('<n>', ' '), output))
    # output = [w.replace('<n>', ' ') for w in output]
    return jsonify(output) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)

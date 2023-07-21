import joblib
import re
import numpy as np
import gradio as gr

def lower_text(text:str):
    return text.lower()

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess(text:str):
    text = lower_text(text)
    text = remove_emoji(text)
    return text.strip()

def load_model():
    vectorizer_filename = 'model/count_vectorizer.joblib'
    model_filename = 'model/bayes.model'
    vectorizer = joblib.load(vectorizer_filename)
    model = joblib.load(model_filename)
    return vectorizer, model

new_text = 'Mong sao thằng pháp không sang xâm lược Việt Nam là may lắm rồi 😂 thắng hay thua kệ bà pháp đi'

vectorizer , model = load_model()

id2label = {
    0: "CLEAN",
    1: "OFFENSIVE",
    2: "HATE"
}


def sentiment_analysis(text :str):
    text_preprocess = preprocess(text)

    text_vectorizer = vectorizer.transform([text_preprocess]) 
    label_pred = model.predict_proba(text_vectorizer)[0]
    list_label = ["CLEAN", 'OFFENSIVE', 'HATE']
    result_dict = dict(zip(list_label, label_pred))
    return result_dict
    
# print(sentiment_analysis('dmm'))
demo = gr.Interface(
    fn=sentiment_analysis, 
    inputs=gr.Textbox(placeholder="Enter a positive or negative sentence here..."), 
    outputs="label", 
    interpretation="default",
    examples=[["Tuyệt vời"]])

demo.launch()




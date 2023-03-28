from flask import Flask,request,jsonify,render_template
import pickle
import tensorflow as tf
from numpy import argmax

app=Flask(__name__)
import transformers
model=tf.keras.models.load_model('model.h5',custom_objects={"TFBertModel": transformers.TFBertModel})

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict' , methods=["POST" ])
def predict():
    if request.method == "POST":
        string_text= request.form.get('message')
        print(string_text)
        tokenized_text= tokenizer(
        text=string_text,
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)

        result= model.predict({'input_ids':tokenized_text['input_ids'],'attention_mask':tokenized_text['attention_mask']})
        output = argmax(result[0])
        return render_template('result.html',prediction=output)
    else:
        return jsonify({
            status : False
        })


if __name__ =='__main__':
    app.run(debug=True)
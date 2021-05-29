import flask
from flask import Flask , render_template , url_for , request
import tensorflow.keras
from tensorflow.keras.models import load_model
import base64
import numpy as np
import cv2
import tensorflow as tf

init_Base64 = 21

label_dict ={0 : "APPLE" , 1 : "BANANA" , 2 : "BIRD" , 3 : "CAMEL" , 4 : "CAR" , 5 : "CAT" , 6 : "ELEPHANT" , 7 : "HORSE" , 8 : "GUITAR" , 9 : "MOUNTAIN" , 10 : "UMBRELLA"}

graph = tf.compat.v1.get_default_graph()

model = load_model("model2_cnn.h5")

app = flask.Flask(__name__ , template_folder='templates')

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict' , methods = ['POST'])
def predict():
  if request.method == 'POST':
    final_pred = None

    draw = request.form['url']

    draw = draw[init_Base64:]

    draw_decode = base64.b64decode(draw)

    image = np.asarray(bytearray(draw_decode) , dtype="uint8")
    image = cv2.imdecode(image , cv2.IMREAD_GRAYSCALE)

    resized = cv2.resize(image , (28,28) , interpolation=cv2.INTER_AREA)
    vect = np.asarray(resized , dtype="uint8")

    vect = vect.reshape(1,28,28 , 1).astype('float32')

    my_pred = model.predict(vect)

    index = np.argmax(my_pred[0])

    final_pred = label_dict[index]

  return render_template('results.html' , prediction = final_pred)

if __name__=='__main__':
  app.run(debug=True)


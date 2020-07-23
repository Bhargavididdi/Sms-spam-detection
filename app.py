from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'C:\Users\hp\Desktop\spam dection intern\code folder\flask\CountVectorizer','rb') as file:
    
   cv =pickle.load(file)
cl=load_model(r'C:\Users\hp\Desktop\spam dection intern\code folder\flask\smsmodel.h5')
cl.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__,template_folder='/Users/hp/Desktop/spam detection intern/code folder/flask/templates',static_folder='/Users/hp/Desktop/spam detection intern/code folder/flask/static')
@app.route('/')
def home():
   return render_template("index.html")
@app.route('/login', methods = ['GET','POST'])
def login():
    if request.method == 'GET':
      img_url = url_for('static',filename = 's.png')
      return render_template("index.html",url=img_url)
    if request.method == 'POST': 
      p = request.form['SMS']
      entered_input = p
      x_intent=cv.transform([entered_input])
      with graph.as_default():
           y_pred=cl.predict(x_intent)
    if(y_pred>0.5):
       return render_template("index.html",showcase="spam")
    else:
         
      return render_template("index.html",showcase="not spam")
if __name__ == '__main__':
    app.run(debug = False)
 
 
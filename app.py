from flask import Flask, render_template, request 
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
from tensorflow.keras.models import model_from_json 
import numpy as np 
import requests
import os
import time
import uuid
import base64 

#Load CNN model and weights
img_width, img_height = 224, 224
json_file = open('./models/model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model_weights_path = 'models/weights.h5'
model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


# Process image and predict label'
def predict(file):
	x=load_img(file, target_size=(img_width,img_height)) # load an image in PIL format (widh X height X channels)
	x=img_to_array(x) #convert to numpy format 
	x = x/255.0 
	x = np.expand_dims(x, axis=0) #convert from NumPy format to Batch format (batchsize x height x weight x channels)
	array = model.predict(x)
	result = array[0][0]
	result = (result > 0.5).astype(np.int)
	return result 


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

# Allowed files 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Initializing flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
   return render_template('template.html', label='', imagesource='../uploads/template.jpg') 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = file.filename

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
                label = 'This image has an anomaly/damage'
            elif result == 1:
                label = 'This image has no anomaly (clean)'			
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True)

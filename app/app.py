from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
Flatten, Dense, Activation, Dropout,LeakyReLU
from PIL import Image
from tensorflow.keras.preprocessing import image




app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 4
animal_classes = ["cat", "dear", "dog", "horse"]

# [Routing

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
    hasil_prediksi  = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
    if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
        file_ext        = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
        if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
            img = image.load_img('.' + gambar_prediksi, target_size=(224, 224))
            
            # Konversi Gambar ke Array
            img = image.img_to_array(img)
            img_arr = np.expand_dims(img, axis=0)
            img_net = tf.keras.applications.mobilenet.preprocess_input(img_arr)
            
			
			# Prediksi Gambar
            predicted = model.predict(img_net)
            label = np.argmax(predicted, axis=1)
			
            hasil_prediksi = animal_classes[label[0]]
			
			# Return hasil prediksi dengan format JSON
            return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
        else:
			# Return hasil prediksi dengan format JSON
            gambar_prediksi = '(none)'
            return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
	

if __name__ == '__main__':
	
    model = load_model("../model/mobileNet3.h5")

	# Run Flask di localhost 
    app.run(host="localhost", port=5000, debug=True)
	
	



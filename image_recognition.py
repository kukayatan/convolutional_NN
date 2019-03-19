
from keras.models import model_from_json
from keras.preprocessing import image
from pathlib import Path
import numpy as np


# Loading json and creating the model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Loading weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Applying the loaded model to the pictures of my dog .. and placing it 
# into the empty dictionary and finally printing the results out in txt file
results_x = {}
basepath = Path('dataset/single_prediction/')
files = basepath.iterdir()
for item in files:
    if item.is_file():
        test_image = image.load_img("dataset/single_prediction/{}".format(item.name), target_size = (32, 32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        if result[0][0] == 1:
            results_x[item.name] = "dog"
        else:
            results_x[item.name] = "cat"

print(results_x)

with open("result.txt","w") as f:
    print(results_x, file=f)

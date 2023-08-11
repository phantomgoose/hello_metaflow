from keras.models import load_model

loaded_model = load_model("output/model.h5")
print(loaded_model.predict([69]))

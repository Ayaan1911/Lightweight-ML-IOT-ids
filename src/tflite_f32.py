"""Control: baseline MLP converted to TFLite WITHOUT quantisation, to separate runtime effect (Keras eager vs TFLite) from int8 effect."""
import numpy as np, keras, tensorflow as tf
from .common import *
from . import data, predictors
from .evaluation import score_test

for task in ("binary", "multi"):
    mid = f"mlp_{task}_tflite_f32"
    if (RES / f"metrics_{mid}.json").exists():
        continue
    m = keras.models.load_model(MOD / f"mlp_{task}.keras", compile=False)
    (MOD / f"{mid}.tflite").write_bytes(tf.lite.TFLiteConverter.from_keras_model(m).convert())
    predictors.register(mid, "tflite", task, [f"{mid}.tflite"], scaler=True)
    X, y = data.load("test", task)
    score_test(mid, task, y, predictors.load(mid)(X.to_numpy(np.float32)), {"note": "float32 TFLite control"})

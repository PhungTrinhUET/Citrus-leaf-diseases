import os, time, numpy as np, tensorflow as tf

def get_model_size(path_h5):
    """Trả về kích thước file (.h5) – MB."""
    return os.path.getsize(path_h5) / (1024 * 1024)

def get_inference_time(model, n_runs=30, input_shape=(224, 224, 3)):
    dummy = np.random.rand(1, *input_shape).astype('float32')
    # warm-up
    _ = model.predict(dummy)
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict(dummy, verbose=0)
    return (time.time() - start) / n_runs * 1000  # ms

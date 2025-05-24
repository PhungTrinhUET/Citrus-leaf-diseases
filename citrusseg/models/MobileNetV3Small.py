import tensorflow as tf
from tensorflow.keras import layers

def build(input_shape=(224, 224, 3), num_classes=5, dropout=0.2):
    """MobileNetV3-Small gốc + GAP + Dense."""
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(base.input, outputs, name='MBV3_Small_Base')

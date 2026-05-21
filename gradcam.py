import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# LAST CONVOLUTION LAYER
last_conv_layer_name = "conv5_block16_concat"

# GRAD MODEL
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# LOAD IMAGE
sample_image_path = test_generator.filepaths[0]
print(f"\nUsing Image: {sample_image_path}")
img = tf.keras.preprocessing.image.load_img(sample_image_path,target_size=(IMG_SIZE, IMG_SIZE))

img_array = tf.keras.preprocessing.image.img_to_array(img)
original_img = img_array.astype(np.uint8)

# PREPROCESS
processed_img = custom_preprocess(img_array.copy())
processed_img = np.expand_dims(processed_img, axis=0)

# COMPUTE GRADIENTS
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(processed_img)

    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

# CHANNEL IMPORTANCE
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# REMOVE BATCH DIMENSION
conv_outputs = conv_outputs[0]

# HEATMAP
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

# RELU + NORMALIZE
heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

# CONVERT TO NUMPY
heatmap = heatmap.numpy()

# RESIZE
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

# APPLY COLORMAP
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# OVERLAY
superimposed_img = cv2.addWeighted(
    cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR),
    0.6,
    heatmap,
    0.4,
    0
)

# DISPLAY
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original X-Ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n===== GRAD-CAM GENERATED SUCCESSFULLY =====")
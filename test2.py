import onnxruntime
import numpy as np
import cv2
import time

# Function to preprocess image
def preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Load ONNX model
session = onnxruntime.InferenceSession('yolov8x-doclaynet.onnx')

# List of sample images to process
img_list = [f'worldhealthstatistics/page_{i}.png' for i in range(1, 11)]

start = time.time()

# Process each image
for image_path in img_list:
    input_image = preprocess_image(image_path, 640)  # Assuming 640 is the input size for your model
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image})

    # Process outputs (you'll need to adapt this part based on your model's specific output format)
    # For example, to draw bounding boxes, you'll need to decode the output tensors into bounding box coordinates

end = time.time()

print("Time taken for processing: ", end - start)

from ultralytics import YOLO
import time
# List of sample images to process
img_list = [f'worldhealthstatistics/page_{i}.png' for i in range(1, 20)]

# Load the document segmentation model

docseg_model = YOLO('https://huggingface.co/DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet/resolve/main/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt?download=true',task="detect")
start=time.time()
# Process the images with the model
results = docseg_model(source=img_list, save=True, show_labels=True, show_conf=True, show_boxes=True)
end=time.time()
print("Time taken for processing: ",end-start)
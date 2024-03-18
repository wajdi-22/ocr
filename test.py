from ultralytics import YOLO
import time
from multiprocessing import Pool

# List of sample images to process
img_list = [f'worldhealthstatistics/page_{i}.png' for i in range(1,11)]


# Load the document segmentation model
docseg_model = YOLO('https://huggingface.co/DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet/resolve/main/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt?download=true',task="detect")

def process_image(img):
    # Process the image with the model
    result = docseg_model(source=img, save=True, show_labels=True, show_conf=True, show_boxes=True)
    return result

if __name__ == "__main__":
    start=time.time()

    # Create a pool of worker processes
    with Pool() as p:
        # Apply the function to each image in the list in parallel
        results = p.map(process_image, img_list)

    end=time.time()
    print("Time taken for processing: ",end-start)
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ============== CONFIGURATION ==============
# Fruits and vegetables that YOLO can detect
PRODUCE_CLASSES = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot'
]

# Plastic/water bottles
BOTTLE_CLASSES = [
    'bottle'
]

# Baked goods / snacks (YOLO detects cake and donut, which covers cookies/pastries)
SNACK_CLASSES = [
    'cake', 'donut'  # These will also catch cookies/pastries visually similar
]

# Sandwiches and hot dogs
SANDWICH_CLASSES = [
    'sandwich', 'hot dog'
]

# Pizza
PIZZA_CLASSES = [
    'pizza'
]

# All detectable classes (removed unreliable chip bag classes)
ALL_CLASSES = PRODUCE_CLASSES + BOTTLE_CLASSES + SNACK_CLASSES + SANDWICH_CLASSES + PIZZA_CLASSES

# Freshness thresholds
FRESH_THRESHOLD = 70  # Above this = Fresh
MODERATE_THRESHOLD = 40  # Above this = Moderate, below = Spoiling

# Colors (BGR format)
COLOR_FRESH = (0, 255, 0)      # Green
COLOR_MODERATE = (0, 165, 255)  # Orange
COLOR_SPOILING = (0, 0, 255)    # Red
COLOR_BOTTLE = (255, 191, 0)    # Deep sky blue for bottles
COLOR_SNACK = (203, 192, 255)   # Pink for baked goods/snacks
COLOR_SANDWICH = (0, 255, 255)  # Yellow for sandwiches
COLOR_PIZZA = (0, 128, 255)     # Orange-red for pizza

# ============== FRESHNESS MODEL ==============
class FreshDetector(torch.nn.Module):
    """Neural network for detecting produce freshness."""
    
    def __init__(self, dropout_rate=0.5):
        super(FreshDetector, self).__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        
        self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def load_freshness_model(model_path="./model/ripe_detector.pth"):
    """Load the freshness detection model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FreshDetector()
    
    if model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, device, transform


def get_freshness_score(image_bgr, model, device, transform):
    """
    Get freshness score for an image.
    
    Returns:
        float: Freshness percentage (0-100, where 100 = perfectly fresh)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Preprocess
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
    
    return probability * 100


def get_freshness_status(score):
    """Convert freshness score to human-readable status."""
    if score >= FRESH_THRESHOLD:
        return "FRESH", COLOR_FRESH, f"Good for ~{int((score-50)/10 + 3)} days"
    elif score >= MODERATE_THRESHOLD:
        return "MODERATE", COLOR_MODERATE, f"Use within ~{int((score-30)/10 + 1)} days"
    else:
        return "SPOILING", COLOR_SPOILING, "Use immediately or discard"


# ============== MAIN DETECTION LOOP ==============
def run_detector():
    """Main function to run the produce freshness and bottle detector."""
    
    print("=" * 50)
    print("  FOOD DETECTOR")
    print("  Fruits, Bottles, Snacks, Sandwiches & Pizza")
    print("=" * 50)
    print("\nLoading models...")
    
    # Load YOLO model for object detection (using nano model for speed)
    yolo_model = YOLO("yolov8n.pt")  # Nano model = fastest, less accurate but much faster
    print("✓ YOLO model loaded (nano - optimized for speed)")
    
    # Load freshness detection model
    try:
        fresh_model, device, transform = load_freshness_model()
        print("✓ Freshness model loaded")
        use_freshness = True
    except Exception as e:
        print(f"⚠ Freshness model not found ({e}), will show detection only")
        fresh_model, device, transform = None, None, None
        use_freshness = False
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for Windows (faster)
    
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Please check that your camera is connected and not in use.")
        return
    
    # Set camera resolution (larger) and optimize for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for less lag
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened at {actual_width}x{actual_height}")
    print("\n" + "=" * 50)
    print("  Press 'q' to quit")
    print("=" * 50 + "\n")
    
    # Frame skip counter for performance
    frame_count = 0
    skip_frames = 2  # Process every 2nd frame for detection (display all)
    last_detections = []  # Cache last detections
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip the frame (fix upside-down camera)
        frame = cv2.flip(frame, -1)  # -1 flips both horizontally and vertically (180 degree rotation)
        
        frame_count += 1
        
        # Only run detection every N frames to reduce lag
        if frame_count % skip_frames == 0:
            # Run YOLO detection with smaller input size for speed
            results = yolo_model.predict(frame, conf=0.5, verbose=False, imgsz=480)
            last_detections = results
        else:
            results = last_detections
        
        # Only print detections on frames where we actually ran detection
        should_print = (frame_count % skip_frames == 0)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            class_names = yolo_model.names
            
            for box in boxes:
                cls_id = int(box.cls.item())
                class_name = class_names[cls_id]
                confidence = box.conf.item()
                
                # Check if it's a detectable class
                if class_name.lower() not in ALL_CLASSES:
                    continue
                    
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Handle bottles separately (no freshness scoring)
                if class_name.lower() in BOTTLE_CLASSES:
                    color = COLOR_BOTTLE
                    label = f"BOTTLE ({confidence:.0%})"
                    sublabel = "Plastic/Water Bottle"
                    if should_print:
                        print(f"Detected: BOTTLE ({confidence:.0%})")
                
                # Handle snacks/baked goods (cookies, cake, donut)
                elif class_name.lower() in SNACK_CLASSES:
                    color = COLOR_SNACK
                    # Display as cookie/snack for user-friendliness
                    display_name = "COOKIE/SNACK" if class_name.lower() in ['cake', 'donut'] else class_name.upper()
                    label = f"{display_name} ({confidence:.0%})"
                    sublabel = "Baked Good"
                    if should_print:
                        print(f"Detected: {display_name} ({confidence:.0%})")
                
                # Handle sandwiches
                elif class_name.lower() in SANDWICH_CLASSES:
                    color = COLOR_SANDWICH
                    label = f"SANDWICH ({confidence:.0%})"
                    sublabel = "Sandwich"
                    if should_print:
                        print(f"Detected: SANDWICH ({confidence:.0%})")
                
                # Handle pizza
                elif class_name.lower() in PIZZA_CLASSES:
                    color = COLOR_PIZZA
                    label = f"PIZZA ({confidence:.0%})"
                    sublabel = "Pizza"
                    if should_print:
                        print(f"Detected: PIZZA ({confidence:.0%})")
                
                # Handle produce with freshness scoring
                elif class_name.lower() in PRODUCE_CLASSES:
                    # Crop the detected produce
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.size > 0 and use_freshness:
                        # Get freshness score
                        freshness = get_freshness_score(cropped, fresh_model, device, transform)
                        status, color, time_est = get_freshness_status(freshness)
                        
                        # Create label
                        label = f"{class_name.upper()}: {status} ({freshness:.0f}%)"
                        sublabel = time_est
                        if should_print:
                            print(f"Detected: {class_name} | Freshness: {freshness:.0f}% | {status} | {time_est}")
                    else:
                        # No freshness model, just show detection
                        color = (255, 255, 0)  # Cyan
                        label = f"{class_name.upper()} ({confidence:.0%})"
                        sublabel = ""
                        if should_print:
                            print(f"Detected: {class_name}")
                else:
                    continue
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw sublabel if exists
                if sublabel:
                    cv2.putText(frame, sublabel, (x1 + 5, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add instructions overlay
        cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Food Detector', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetector stopped.")


if __name__ == "__main__":
    run_detector()

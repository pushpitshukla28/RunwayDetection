Hybrid Deep Learning for Robust Runway Detection

Honeywell Hackspace Submission

This project is Team Starvaders' submission for the Honeywell Hackspace hackathon, organized in collaboration with Manipal Institute of Technology, Bengaluru.

We developed a robust computer vision pipeline to accurately detect and outline airport runways from complex aerial imagery. The system is designed to be resilient to challenging conditions, such as noise, poor visibility, and ambiguous backgrounds, which are critical for enhancing aviation safety.

The Challenge

The goal is to detect runways and output their precise coordinates. A simple segmentation model can fail in real-world scenarios, producing noisy masks or incorrectly identifying other structures (like taxiways or roads) as runways. This results in extremely low "Anchor" (polygon) and "Boolean" (geometric) scores.

Our Solution: The "Seek and Verify" Pipeline

To solve this, we developed a novel hybrid pipeline that uses two models to cross-validate each other. This "Seek and Verify" logic allows the system to intelligently filter out noise and lock onto the correct runway.

Segmentation Model (U-Net + ResNet34): A segmentation model first predicts a raw mask of all potential runway-like objects. This is fast but can be noisy.

Keypoint Model (U-Net + ResNet34): A second model, trained in parallel, predicts the sub-pixel coordinates of the 6 keypoints of the runway (four corners and two centerline points).

"Seek and Verify" Logic:

The raw mask is broken into its separate connected components (blobs).

The predicted keypoints are used as "votes." The system finds the specific blob that contains the most keypoints.

This blob is isolated as the verified runway, and all other noise is discarded.

Final Output: Precise polygon coordinates are then extracted from this clean, verified mask, resulting in high-fidelity object detection.

This approach is designed to transform noisy, ambiguous masks into precise coordinates, dramatically improving anchor and boolean scores in difficult images.

Tech Stack

Framework: PyTorch

Models: segmentation-models-pytorch (U-Net w/ ResNet34 backbone)

Data Augmentation: Albumentations

Core CV: OpenCV, NumPy

Evaluation: Pandas

Evaluation

We built a comprehensive evaluation script (final_evaluation.py) that measures three key metrics to determine real-world performance:

Mask IoU: Standard Intersection over Union of the verified mask against the ground truth mask.

Anchor Score: The Polygon IoU between the final extracted runway polygon and the ground truth polygon. This is the primary metric for coordinate accuracy.

Boolean Score: A custom logic that returns 1.0 if the predicted runway's centerline is correctly positioned between the runway edges, and 0.0 otherwise.

How to Run

Clone the repository:

git clone [URL-to-your-repo]
cd [repo-name]


Install dependencies:

pip install torch torchvision segmentation-models-pytorch opencv-python-headless albumentations numpy pandas tqdm


Update Paths:
Open final_evaluation.py and update the variables in the Configuration section (lines 13-17) to point to your local data and model files:

TEST_IMG_DIR = "path/to/your/test/images"
ANNOTATIONS_JSON_PATH = "path/to/your/test_labels.json"
SEG_MODEL_PATH = "path/to/your/best_segmentation_model.pth"
KP_MODEL_PATH = "path/to/your/runway_keypoint_model.pth"


Run the evaluation:

python final_evaluation.py


Check results:

The final mean scores will be printed to the console.

A detailed evaluation_metrics_final.csv will be saved in the final_evaluation_results folder.

Side-by-side visualization images will be saved in final_evaluation_results/visualizations.

Team Starvaders

Pushpit Shukla

Nandini Binani

Kush Khanna

Aditi Ojha

Acknowledgments

We would like to thank Honeywell Technology Solutions and Manipal Institute of Technology, Bengaluru for organizing this hackathon and providing a challenging, real-world problem to solve.

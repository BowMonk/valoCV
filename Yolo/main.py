import cv2
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from dataclasses_json import dataclass_json
from supervision import Detections


@dataclass_json
@dataclass
class COCOCategory:
	id: int
	name: str
	supercategory: str


@dataclass_json
@dataclass
class COCOImage:
	id: int
	width: int
	height: int
	file_name: str
	license: int
	date_captured: str
	coco_url: Optional[str] = None
	flickr_url: Optional[str] = None


@dataclass_json
@dataclass
class COCOAnnotation:
	id: int
	image_id: int
	category_id: int
	segmentation: List[List[float]]
	area: float
	bbox: Tuple[float, float, float, float]
	iscrowd: int


@dataclass_json
@dataclass
class COCOLicense:
	id: int
	name: str
	url: str


@dataclass_json
@dataclass
class COCOJson:
	images: List[COCOImage]
	annotations: List[COCOAnnotation]
	categories: List[COCOCategory]
	licenses: List[COCOLicense]


def load_coco_json(json_file: str) -> COCOJson:
	import json

	with open(json_file, "r") as f:
		json_data = json.load(f)

	return COCOJson.from_dict(json_data)


class COCOJsonUtility:
	def get_image_by_path(coco_data: COCOJson, image_path: str) -> Optional[COCOImage]:
		for image in coco_data.images:
			if image.file_name == image_path:
				return image
		return None

	@staticmethod
	def get_annotations_by_image_id(coco_data: COCOJson, image_id: int) -> List[COCOAnnotation]:
		return [annotation for annotation in coco_data.annotations if annotation.image_id == image_id]

	@staticmethod
	def get_annotations_by_image_path(coco_data: COCOJson, image_path: str) -> Optional[List[COCOAnnotation]]:
		image = COCOJsonUtility.get_image_by_path(coco_data, image_path)
		if image is not None:
			return COCOJsonUtility.get_annotations_by_image_id(coco_data, image.id)
		return None

	@staticmethod
	def annotations2detections(annotations: List[COCOAnnotation]) -> Detections:
		bbox_list = [annotation.bbox for annotation in annotations]
		class_id = [annotation.category_id for annotation in annotations]
		xyxy = np.array([[x_min, y_min, x_min + width, y_min + height] for (x_min, y_min, width, height) in bbox_list],
						dtype=int)

		return Detections(
			xyxy=xyxy,
			class_id=np.array(class_id, dtype=int)
		)


def point_in_box(point, box):
	x, y = point
	x1, y1, x2, y2 = box
	return x1 <= x <= x2 and y1 <= y <= y2


def main():
	coco_data = load_coco_json(json_file='./bot3/trainval.json')

	identified_enemies = {}
	# Load your custom YOLOv5m model
	model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

	# Open the video capture
	cap = cv2.VideoCapture('yolov5/videos/bot.mp4')  # Replace with your video path

	fps = cap.get(cv2.CAP_PROP_FPS)

	counter = 0

	# Original dimensions of the image
	width_orig = 1280  # replace with your value
	height_orig = 720  # replace with your value

	# New dimensions
	width_new = 640
	height_new = 640

	while cap.isOpened():

		annotations_test = COCOJsonUtility.get_annotations_by_image_path(coco_data=coco_data,
																		 image_path=str(counter) + ".jpg")
		try:
			ground_truth = COCOJsonUtility.annotations2detections(annotations=annotations_test)
		except:
			ground_truth = Detections(
				xyxy=np.empty((0, 4), dtype=np.float32),
				confidence=np.array([], dtype=np.float32),
				class_id=np.array([], dtype=int)
			)
		ground_truth.class_id = ground_truth.class_id - 1

		# Read the frame
		ret, frame = cap.read()

		if not ret:
			break
		#
		frame = cv2.resize(frame, (640, 640))

		# cv2.putText(frame, 'FPS: ' + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (124, 252, 0), 2, cv2.LINE_AA)

		labels = ['boomybot', 'enemyBody', 'enemyHead', 'smoke', 'splat', 'teammate', 'walledoff enemy']

		# Perform inference
		results = model(frame)

		# Get bounding box coordinates, labels, and confidence scores
		results.pred = [x.to('cpu').numpy() for x in results.pred]
		confidence_threshold = 0.5  # Set your confidence threshold here

		center_x, center_y, class_id_gt = None, None, None

		if(len(ground_truth) > 0):
			xyxy = ground_truth.xyxy[0]  # Get the first bounding box
			class_id_gt = ground_truth.class_id[0]  # Get the first class ID

			# Scaling
			xyxy = np.array(xyxy) * np.array(
				[width_new / width_orig, height_new / height_orig, width_new / width_orig, height_new / height_orig])
			box = [int(point) for point in xyxy]
			class_id_gt = int(class_id_gt)
			x1, y1, x2, y2 = box
			if class_id_gt == 0:
				class_id_gt = 2

			cv2.circle(frame, (x1, y1), radius=6, color=(255, 0, 0),
					   thickness=2)  # Increased radius and changed color to blue
			# cv2.putText(frame, f'{label}:', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

			x1, y1, x2, y2 = [int(coord) for coord in xyxy]
			center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2



		last_label_y = None

		for *box, confidence, class_id in results.pred[0]:
			if confidence >= confidence_threshold:  # Filter out low-confidence detections
				box = [int(point) for point in box]
				class_id = int(class_id)
				x1, y1, x2, y2 = box
				label = labels[class_id]



					# Choose the position of the label based on the last label
				if last_label_y is not None and abs(y1 - last_label_y) < 20:
					label_y = last_label_y - 20
				else:
					label_y = y1 - 10
				last_label_y = label_y

				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
							(0, 255, 0), 2)


				if center_y is not None:
					if point_in_box((center_x, center_y), box) and class_id_gt == class_id:
							identified_enemies[class_id_gt] = True
							break


		cv2.imshow('YOLOv5 Video', frame)
		counter += 1

		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

	correct_detection_count = sum(identified_enemies.values())
	total_enemies = len(identified_enemies)
	accuracy = correct_detection_count / total_enemies if total_enemies > 0 else 0
	print(f"Accuracy of head detection for distinct enemies: {accuracy}")


# # Compute the delay in frames between head and body detection
# for identifier in enemy_body_frames.keys():
# 	if identifier in enemy_head_frames.keys():
# 		delay = enemy_body_frames[identifier] - enemy_head_frames[identifier]
# 		print(f'For enemy {identifier}, the delay is {delay} frames.')
#


if __name__ == '__main__':
	main()

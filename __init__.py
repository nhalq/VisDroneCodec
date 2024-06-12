import json
import os
import pathlib
import shutil

from PIL import Image
from tqdm import tqdm


class VisDroneCodec:
    # Note: 0 is ignored regions (background)
    CATEGORIES = ['pedestrian',       # 1
                  'people',           # 2
                  'bicycle',          # 3
                  'car',              # 4
                  'van',              # 5
                  'truck',            # 6
                  'tricycle',         # 7
                  'awning-tricycle',  # 8
                  'bus',              # 9
                  'motor',            # 10
                  'others']           # 11

    def __init__(self, path: str) -> None:
        self.source = pathlib.Path(path)

    def coco_create_categories(self):
        return [
            dict({
                'id': i,
                'name': name,
                'supercategory': 'none'
            }) for i, name in enumerate(VisDroneCodec.CATEGORIES, start=1)
        ]

    def coco_create_object(self):
        self.coco = dict({
            'info': dict(),
            'licenses': list(),
            'images': list(),
            'annotations': list(),
            'categories': self.coco_create_categories()
        })

        return self.coco

    def coco_add_image(self, image_path: pathlib.Path):
        image_id = len(self.coco['images'])
        width, height = Image.open(image_path).size
        self.coco['images'].append({
            'id': image_id,
            'file_name': image_path.name,
            'width': width,
            'height': height
        })

        return image_id

    def coco_add_annotation(self, image_id: int, bbox, category: int):
        _, _, width, height = bbox

        annotation_id = len(self.coco['annotations'])
        self.coco['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category,
            'bbox': bbox,
            'area': width * height,
            'size': [width, height],
            'iscrowd': 0,
        })

        return annotation_id

    def coco_export(self, path: str):
        export_path = pathlib.Path(path)
        export_images_path = export_path / 'images'
        export_annotations_path = export_path / 'annotations'
        os.makedirs(export_images_path, exist_ok=True)
        os.makedirs(export_annotations_path, exist_ok=True)

        coco = self.coco_create_object()

        import_images_path = self.source / 'images'
        import_annotations_path = self.source / 'annotations'

        annotations_glob = import_annotations_path.glob('*.txt')
        for annotation_path in tqdm(annotations_glob, desc='Converting'):
            name = annotation_path.with_suffix('.jpg').name
            image_path = import_images_path / name

            shutil.copy(image_path, export_images_path)
            image_id = self.coco_add_image(image_path)

            with open(annotation_path, 'r') as fs:
                for row in map(lambda s: s.strip().split(','), fs.readlines()):
                    bb_left, bb_top, width, height = map(int, row[:4])
                    category = int(row[5])

                    bbox = [bb_left, bb_top, max(1, width), max(1, height)]
                    self.coco_add_annotation(image_id, bbox, category)

        with open(export_annotations_path / 'instances.json', 'w') as fs:
            json.dump(coco, fs)

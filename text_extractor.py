import gc
import os
from pathlib import Path
from typing import Optional

import torch
from mmocr.utils import stitch_boxes_into_lines
from mmocr.utils.ocr import MMOCR


class TextExtractor:
    def __init__(self,
                 model_config_dir: str = os.path.join(str(Path.cwd()), 'configs/'),
                 batch_mode: bool = True,
                 det: str = 'FCE_CTW_DCNv2',
                 recog: str = 'MASTER',
                 det_config: Optional[str] = None,
                 det_ckpt: Optional[str] = None,
                 recog_config: Optional[str] = None,
                 recog_ckpt: Optional[str] = None):
        """
        :param model_config_dir: 'configs' path of MMOCR, required to start MMOCR object. 
        """
        self.batch_mode = batch_mode
        self.ocr = MMOCR(det=det,
                         recog=recog,
                         config_dir=model_config_dir,
                         batch_mode=batch_mode,
                         det_config=det_config,
                         det_ckpt=det_ckpt,
                         recog_ckpt=recog_ckpt,
                         recog_config=recog_config)

    @torch.no_grad()
    def extract_text(self,
                     image):
        """
        Extract texts from given image path
        :param image: img path
        :return: (ocr_text)
        """
        torch.cuda.empty_cache()
        gc.collect()
        results = self.ocr.readtext(image,
                                    print_result=False,
                                    imshow=False,
                                    details=True,
                                    merge=False,
                                    batch_mode=self.batch_mode)
        gc.collect()
        torch.cuda.empty_cache()
        stitched_text = TextExtractor.sort_ocr_results_and_extract_text(results).lower()
        return stitched_text

    @staticmethod
    def sort_ocr_results_and_extract_text(results,
                                          box_stitch_max_x_dist=50,
                                          min_y_overlap_ratio=0.8,
                                          text_score_threshold=0.1):
        boxes = results[0]['result']
        boxes = list(filter(lambda box: box['text_score'] > text_score_threshold, boxes))

        def get_centroid_from_box(box):
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[5]
            return (x1 + x2) / 2, (y1 + y2) / 2

        boxes_in_lines = stitch_boxes_into_lines(boxes,
                                                 max_x_dist=box_stitch_max_x_dist,
                                                 min_y_overlap_ratio=min_y_overlap_ratio)
        sorted_results = sorted(boxes_in_lines, key=lambda x: get_centroid_from_box(x['box'])[1])
        stitched_text = " ".join(list(map(lambda box_line: box_line['text'], sorted_results)))
        return stitched_text

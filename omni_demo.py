import torch
from PIL import Image
import base64
import io
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# 初始化模型（全局只初始化一次）
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)

def process_image():
    # ============= 参数配置区域 =============
    input_image_path = "screenshot.png"      # 输入图片路径
    output_image_path = "labeled_image.png"         # 输出图片路径
    box_threshold = 0.05                     # 检测框置信度阈值
    iou_threshold = 0.1                      # NMS的IOU阈值
    use_paddleocr = True                     # 使用PaddleOCR（False则用EasyOCR）
    imgsz = 640                              # 图标检测分辨率
    # ============= 参数配置结束 =============
    
    # 加载输入图像
    image_input = Image.open(input_image_path)
    
    # 处理参数计算
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # OCR处理
    ocr_bbox_rslt, _ = check_ocr_box(
        image_input,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold':0.9},
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt

    # 图标检测和标注
    labeled_img, _, parsed_content_list = get_som_labeled_img(
        image_input,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz
    )

    # 保存结果图像
    image = Image.open(io.BytesIO(base64.b64decode(labeled_img)))
    image.save(output_image_path)
    print(f"结果图片已保存至：{output_image_path}")

    # 输出解析结果
    print("\n解析到的界面元素：")
    for i, element in enumerate(parsed_content_list):
        print(f"[元素 {i}] {element}")

if __name__ == "__main__":
    process_image()
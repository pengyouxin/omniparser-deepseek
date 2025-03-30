import torch
from PIL import Image
import base64
import io
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from openai import OpenAI
import pyautogui

# 初始化模型（全局只初始化一次）
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)

def process_image():
    # ============= 参数配置区域 =============
    pyautogui.screenshot("screenshot.png")
    input_image_path = "screenshot.png"      # 输入图片路径
    output_image_path = "labeled_image.png"  # 输出图片路径
    box_threshold = 0.05                     # 检测框置信度阈值
    iou_threshold = 0.1                      # NMS的IOU阈值
    use_paddleocr = True                     # 使用PaddleOCR（False则用EasyOCR）
    imgsz = 640                              # 图标检测分辨率
    
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

    # 保存解析结果
    parsed_content_str = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    with open("parsed_content.txt", "w", encoding="utf-8") as f:
        f.write(parsed_content_str)
    print(f"解析结果已保存至：parsed_content.txt")
    return parsed_content_str

def chat(msg):
    client = OpenAI(
        api_key="sk-Z1rQi13iPuwpTZDzihTP93bEPnZ5knphNJIRR6zLf2GXYlZc",
        base_url="https://api.aiclaude.site/v1"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_image",
                "description": "识别屏幕中的元素，解析出各个元素的坐标等信息",
            }
        },
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=msg,
        tools=tools,
        temperature=0
    )
    return response.choices[0].message

if __name__ == "__main__":
    msg = [{"role":"system", "content":"""
    你可以处理用户的请求，例如：识别屏幕中的元素，解析出各个元素的坐标等信息。
    当用户的要求需要操作计算机才能完成时，你需要自主进行屏幕截屏识别元素并执行点击和键盘输入操作
    """}]
    msg.append({"role": "user", "content": "chrome浏览器在屏幕中的坐标是？"})
    message = chat(msg)
    print(message)
    msg.append({"role": message.role, "content": message.content})
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "process_image":
                screenshot_info = process_image()
                msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "process_image", "content": screenshot_info})

    message = chat(msg)
    print(message.content)

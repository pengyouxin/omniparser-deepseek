import torch
from PIL import Image
import base64
import io
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from openai import OpenAI
import pyautogui
from time import sleep
import json

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

def bbox_to_coords(bbox, screen_width, screen_height):
    """将 bbox 坐标转换为屏幕坐标."""
    xmin, ymin, xmax, ymax = bbox

    # 考虑 Mac 顶部菜单栏的偏移
    menu_bar_height = 25

    # 向上偏移以避免点击到文件名
    y_offset = -15  # 向上偏移15像素

    # 计算相对坐标
    x_center = int((xmin + xmax) / 2 * screen_width)
    y_center = int((ymin + ymax) / 2 * (screen_height - menu_bar_height)) + menu_bar_height + y_offset

    # 添加调试信息
    # print(f"\n坐标转换详情:")
    # print(f"屏幕尺寸: {screen_width} x {screen_height}")
    # print(f"原始bbox: {bbox}")
    # print(f"x轴变换: {xmin:.4f} -> {xmax:.4f} 中点: {(xmin + xmax) / 2:.4f}")
    # print(f"y轴变换: {ymin:.4f} -> {ymax:.4f} 中点: {(ymin + ymax) / 2:.4f}")
    # print(f"考虑菜单栏偏移: {menu_bar_height}px")
    # print(f"向上偏移: {y_offset}px")
    # print(f"计算结果: x={x_center}, y={y_center}")

    # 确保坐标在屏幕范围内
    x_center = max(0, min(x_center, screen_width))
    y_center = max(0, min(y_center, screen_height))

    return x_center, y_center

def click(bbox):
    """点击指定的 bbox."""
    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    print(f"当前屏幕分辨率: {screen_width}x{screen_height}")

    # 获取点击坐标
    x, y = bbox_to_coords(bbox, screen_width, screen_height)

    print(f"目标坐标: x={x}, y={y}")
    sleep(0.5)

    # 移动鼠标到指定位置（使用缓动效果）
    pyautogui.moveTo(x, y, duration=0.5, tween=pyautogui.easeOutQuad)
    sleep(0.5)

    # 获取当前鼠标位置以验证
    current_x, current_y = pyautogui.position()
    print(f"当前鼠标位置: x={current_x}, y={current_y}")

    # 点击鼠标
    pyautogui.click()
    print(f"已点击坐标: x={x}, y={y}")

def type_text(bbox, text):
    click(bbox)
    pyautogui.typewrite(text)
    pyautogui.press("enter")
    print(f"已输入文本: {text}")

def chat(msg):
    client = OpenAI(
        # api_key="sk-Lfcrp0IWFYzPU5a6H34RkmqjuEJ63cbWIoC7G1aoou4VTuh7",
        # api_key="sk-Pm9VAO5FRnQRS9W3mCskCzTpP1kJ52E5w3RgXxlBDVwNmW3n",
        # base_url="https://chataiapi.com/v1"
        api_key="sk-974346a3d84b49f2819e07f67dd9efef",
        base_url="https://api.deepseek.com/v1"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_image",
                "description": "识别屏幕中的元素，解析出各个元素的坐标等信息，比如bbox坐标、icon类型、icon内容等。",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "在bbox坐标处点击鼠标",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "description": "bbox坐标，格式为[xmin, ymin, xmax, ymax]"
                        }
                    },
                "required": ["bbox"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "type_text",
                "description": "在bbox坐标处输入文本并按回车键",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "description": "bbox坐标，格式为[xmin, ymin, xmax, ymax]"
                        },
                        "text": {
                            "type": "string",
                            "description": "要输入的文本"
                        }
                    },
                "required": ["bbox", "text"]
                }
            }
        }
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
    你能够根据用户请求，分析出执行步骤，然后根据步骤执行相应的操作吗。
    例如：
    用户：在chrome浏览器中搜索xxx
    你：step1：截屏分析屏幕元素的坐标。step2：找到chrome图标。step3：点击chrome图标。step4：截屏识别出搜索框。step5：在搜索框中输入xxx。step5：按回车键。
    """}]
    user_input = input("user:")
    msg.append({"role": "user", "content": user_input})
    message = chat(msg)
    print(message)
    # msg.append({"role": message.role, "content": message.content})
    # msg.append(message)
    """添加历史message否则会导致以下报错
    openai.BadRequestError: Error code: 400 - {'error': {'message': "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
    """

    # if message.tool_calls:
    #     for tool_call in message.tool_calls:
    #         if tool_call.function.name == "process_image":
    #             screenshot_info = process_image()
    #             msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "process_image", "content": screenshot_info})
    #     message = chat(msg)
    #     print(message)
    #     msg.append(message)

    # if message.tool_calls:
    #     for tool_call in message.tool_calls:
    #         if tool_call.function.name == "click":
    #             args = json.loads(tool_call.function.arguments)
    #             click(args.get("bbox"))
    #             msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "click", "content": "已点击坐标"})
    #     message = chat(msg)
    #     print(message)
    #     msg.append(message)

    # if message.tool_calls:
    #     for tool_call in message.tool_calls:
    #         if tool_call.function.name == "type_text":
    #             args = json.loads(tool_call.function.arguments)
    #             type_text(args.get("bbox"), args.get("text"))
    #             msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "type_text", "content": f"已输入文本: {args.get('text')}"})
    #     message = chat(msg)
    #     print(message)
    #     msg.append(message)

    while message.tool_calls:
        msg.append(message)
        for tool_call in message.tool_calls:
            if tool_call.function.name == "process_image":
                screenshot_info = process_image()
                msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "process_image", "content": screenshot_info})
            if tool_call.function.name == "click":
                args = json.loads(tool_call.function.arguments)
                click(args.get("bbox"))
                msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "click", "content": "已点击坐标"}) 
            if tool_call.function.name == "type_text":
                args = json.loads(tool_call.function.arguments)
                type_text(args.get("bbox"), args.get("text"))
                msg.append({"role": "tool", "tool_call_id": tool_call.id, "name": "type_text", "content": f"已输入文本: {args.get('text')}"})       
        message = chat(msg)
        print(message)
        

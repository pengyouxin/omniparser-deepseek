import requests
from PIL import Image
import base64
import io
import pyautogui
from time import sleep
import json
import ast  # 用于解析字符串形式的字典

def process_image(
        image_path: str,
        api_url: str = "http://127.0.0.1:8000/process_image",
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        use_paddleocr: bool = True,
        imgsz: int = 640
):
    files = {
        'file': ('image.png', open(image_path, 'rb'), 'image/png')
    }

    params = {
        'box_threshold': box_threshold,
        'iou_threshold': iou_threshold,
        'use_paddleocr': use_paddleocr,
        'imgsz': imgsz
    }

    response = requests.post(api_url, files=files, params=params)

    if response.status_code == 200:
        result = response.json()

        if result['status'] == 'success':
            labeled_image = Image.open(io.BytesIO(base64.b64decode(result['labeled_image'])))
            labeled_image.save('labeled_image.png')
            return {
                'status': 'success',
                'labeled_image': labeled_image,
                'parsed_content': result['parsed_content'],
                'label_coordinates': result['label_coordinates']
            }
        else:
            return {'status': 'error', 'message': result.get('message', 'Unknown error')}
    else:
        return {'status': 'error', 'message': f'HTTP error {response.status_code}'}

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
    print(f"\n坐标转换详情:")
    print(f"屏幕尺寸: {screen_width} x {screen_height}")
    print(f"原始bbox: {bbox}")
    print(f"x轴变换: {xmin:.4f} -> {xmax:.4f} 中点: {(xmin + xmax) / 2:.4f}")
    print(f"y轴变换: {ymin:.4f} -> {ymax:.4f} 中点: {(ymin + ymax) / 2:.4f}")
    print(f"考虑菜单栏偏移: {menu_bar_height}px")
    print(f"向上偏移: {y_offset}px")
    print(f"计算结果: x={x_center}, y={y_center}")

    # 确保坐标在屏幕范围内
    x_center = max(0, min(x_center, screen_width))
    y_center = max(0, min(y_center, screen_height))

    return x_center, y_center

def double_click_bbox(bbox):
    """双击指定的 bbox."""
    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    print(f"当前屏幕分辨率: {screen_width}x{screen_height}")

    # 获取点击坐标
    x, y = bbox_to_coords(bbox, screen_width, screen_height)

    print(f"\n即将执行双击:")
    print(f"目标坐标: x={x}, y={y}")
    print("3秒准备时间...")
    sleep(3)

    # 移动鼠标到指定位置
    pyautogui.moveTo(x, y, duration=0.5)

    print("鼠标已就位，1秒后双击...")
    sleep(1)

    # 执行双击
    pyautogui.doubleClick()

    print(f"已双击坐标: x={x}, y={y}")

def click_bbox(bbox):
    """点击指定的 bbox."""
    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    print(f"当前屏幕分辨率: {screen_width}x{screen_height}")

    # 获取点击坐标
    x, y = bbox_to_coords(bbox, screen_width, screen_height)

    print(f"\n即将执行点击:")
    print(f"目标坐标: x={x}, y={y}")
    print("1秒准备时间...")
    sleep(1)

    # 移动鼠标到指定位置（使用缓动效果）
    pyautogui.moveTo(x, y, duration=0.5, tween=pyautogui.easeOutQuad)

    print("鼠标已就位，0.5秒后点击...")
    sleep(0.5)

    # 获取当前鼠标位置以验证
    current_x, current_y = pyautogui.position()
    print(f"当前鼠标位置: x={current_x}, y={current_y}")

    # 点击鼠标
    pyautogui.click()
    print(f"已点击坐标: x={x}, y={y}")

def find_target(target_name:str, icons):
    """在解析内容中查找 target_name 的图标."""
    target_name = target_name.strip().lower()
    for i, icon in enumerate(icons):
        if isinstance(icon, dict) and 'content' in icon:
            content = icon['content'].strip().lower()
            if target_name in content:
                print(f"找到 {target_name}，图标索引: {i}")
                return icon['bbox']
    return None

if __name__ == "__main__":
    # 获取并打印屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    print(f"当前屏幕分辨率: {screen_width}x{screen_height}")

    img = pyautogui.screenshot()
    img.save('screenshot.png')
    image_path = "./screenshot.png"
    result = process_image(
        image_path=image_path,
        box_threshold=0.05,
        iou_threshold=0.1,
        use_paddleocr=True,
        imgsz=640
    )

    if result['status'] == 'success':
        icons = result['parsed_content']
        # target_bbox = find_target('google chrome', icons)
        target_bbox = find_target('microsoft edge ', icons)

        if target_bbox:
            print("找到目标坐标:", target_bbox)
            click_bbox(target_bbox)
        else:
            print("未找到目标图标")
    else:
        print("Error:", result['message'])

# print(f"当前屏幕分辨率: {pyautogui.size()}")


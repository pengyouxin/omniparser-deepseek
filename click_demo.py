import pyautogui
import time

def bbox_to_coords(bbox, screen_width, screen_height):
    """将 bbox 坐标转换为屏幕坐标."""
    xmin, ymin, xmax, ymax = bbox
    x_center = int((xmin + xmax) / 2 * screen_width)
    y_center = int((ymin + ymax) / 2 * screen_height)
    return x_center, y_center

def click_element(element):
    """点击指定的 bbox."""
    bbox = element['bbox']
    screen_width, screen_height = pyautogui.size()
    x, y = bbox_to_coords(bbox, screen_width, screen_height)

    # 移动鼠标到指定位置
    pyautogui.moveTo(x, y, duration=0.5, tween=pyautogui.easeInOutQuad)  # duration 是移动时间，单位为秒

    # 点击鼠标
    pyautogui.click()

    print(f"点击了{element['content']}")

if __name__ == '__main__':

    time.sleep(1)

    # 示例 bbox (来自您提供的数据)
    element = {'type': 'icon', 'bbox': [0.5614868402481079, 0.9512439370155334, 0.5863829255104065, 0.9973732829093933], 'interactivity': True, 'content': 'Microsoft Edge browser', 'source': 'box_yolo_content_yolo'} # chrome

    # 点击 bbox
    click_element(element)
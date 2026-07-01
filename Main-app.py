import gradio as gr
from ultralytics import YOLO
from PIL import Image
import urllib.parse

# Load pre-trained YOLO model
model = YOLO(r'D:\python project\ultralytics-main\runs\detect\train13\weights\best.pt')

# Gesture meaning dictionary
gesture_meanings = {
    "拳头": "人物做出了拳头手势，通常表示力量、决心或愤怒",
    "数字7": "人物比出了数字7，可能是在表示数字7或特定手势",
    "数字1": "人物竖起了食指，通常表示数字1、指向或强调",
    "小拇指": "人物伸出了小拇指，可能有特定文化含义或约定",
    "手势L": "人物比出了L手势，可能表示字母L或特定含义",
    "比耶": "人物比出了V字手势(耶)，通常表示胜利、和平或拍照姿势",
    "数字3": "人物比出了三根手指，可能表示数字3或特定含义",
    "666": "人物比出了六的手势，可能表示数字6或特定文化含义",
    "摇滚手势": "人物比出了Rock手势，通常用于摇滚音乐场景",
    "数字5": "人物张开了手掌，表示数字5或停止"
}

# Gesture to search keywords mapping
gesture_keywords = {
    "拳头": "拳头手势含义 拳头手势文化差异",
    "数字7": "数字7手势含义 数字7手势文化差异",
    "数字1": "数字1手势含义 食指手势文化差异",
    "小拇指": "小拇指手势含义 小拇指手势文化差异",
    "手势L": "L手势含义 L手势文化差异",
    "比耶": "V字手势含义 比耶手势文化差异",
    "数字3": "数字3手势含义 三指手势文化差异",
    "666": "666手势含义 六手势文化差异",
    "摇滚手势": "摇滚手势含义 金属礼手势文化",
    "数字5": "数字5手势含义 手掌手势文化差异"
}


def predict_image(image, conf_threshold, iou_threshold):
    # Perform inference
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=540)

    # Get classification results
    class_results = []
    for r in results:
        # Plot results image
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        # Extract class info
        classNameObj = {
            "A": "拳头", "number 7": "数字7", "D": "数字1", "I": "小拇指",
            "L": "手势L", "V": "比耶", "W": "数字3", "Y": "666",
            "I love you": "摇滚手势", "number 5": "数字5"
        }
        if r.boxes:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                class_results.append(f"{classNameObj[class_name]} ({confidence:.2f})")

    # Remove duplicates and join results
    unique_results = ", ".join(sorted(set(class_results), key=class_results.index)) if class_results else "未检测到目标"
    return im, unique_results


def analyze_gesture(gesture_result):
    # Extract gesture name from result
    gesture_name = gesture_result.split(' (')[0] if ' (' in gesture_result else gesture_result

    # Get gesture meaning
    meaning = gesture_meanings.get(gesture_name, "未知手势，暂无含义解释")
    return meaning


def get_baidu_search_url(gesture_result):
    try:
        # Extract gesture name from result
        gesture_name = gesture_result.split(' (')[0] if ' (' in gesture_result else gesture_result

        if not gesture_name or gesture_name == "未检测到目标":
            return "未检测到有效手势，无法搜索"

        # Get search keywords
        keywords = gesture_keywords.get(gesture_name, f"{gesture_name}手势含义")

        # Safe URL encoding
        encoded_query = urllib.parse.quote_plus(keywords)
        return f"https://www.baidu.com/s?wd={encoded_query}"
    except Exception as e:
        print(f"Error generating search URL: {str(e)}")
        return "生成搜索链接时出错"


# Custom CSS styling
custom_css = """
.analysis-btn {
    background: #4a90e2 !important;
    color: white !important;
    border: none !important;
}
.analysis-btn:hover {
    background: #3a7bc8 !important;
}
.search-btn {
    background: #e74c3c !important;
    color: white !important;
    border: none !important;
}
.search-btn:hover {
    background: #c0392b !important;
}
.result-box {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
}
"""

with gr.Blocks(css=custom_css, title="手势识别系统") as demo:
    gr.Markdown("""<div style="text-align: center;">
        <h1>手势识别系统</h1>
        <p>上传手势图片进行识别，获取手势含义和文化背景</p>
    </div>""")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            with gr.Group():
                image_input = gr.Image(type="pil", label="上传手势图片", height=300)
                with gr.Row():
                    conf_slider = gr.Slider(0, 1, 0.50, label="置信度阈值", interactive=True)
                    iou_slider = gr.Slider(0, 1, 0.50, label="交并比阈值", interactive=True)
                submit_btn = gr.Button("识别手势", variant="primary")

        with gr.Column(scale=1, min_width=400):
            with gr.Group():
                image_output = gr.Image(type="pil", label="识别结果", height=300)
                text_output = gr.Textbox(label="检测结果", interactive=False)

                with gr.Row():
                    analysis_btn = gr.Button("分析手势含义", elem_classes="analysis-btn")
                    search_btn = gr.Button("百度搜索手势", elem_classes="search-btn")

                with gr.Group(visible=False) as result_group:
                    analysis_output = gr.Textbox(label="手势含义", interactive=False)
                    search_output = gr.Textbox(label="搜索链接", interactive=False)

    # Submit button event
    submit_btn.click(
        fn=predict_image,
        inputs=[image_input, conf_slider, iou_slider],
        outputs=[image_output, text_output]
    )

    # Analysis button event
    analysis_btn.click(
        fn=analyze_gesture,
        inputs=[text_output],
        outputs=[analysis_output]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[result_group]
    )

    # Search button event
    search_btn.click(
        fn=get_baidu_search_url,
        inputs=[text_output],
        outputs=[search_output]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[result_group]
    ).then(
        None,
        _js="""
        function(url) {
            if (url.startsWith("http")) {
                window.open(url, '_blank');
            }
            return '';
        }
        """,
        inputs=[search_output],
        outputs=[]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        auth=None,
        auth_message=None,
        prevent_thread_lock=False,
        show_error=True,
        debug=False
    )
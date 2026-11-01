
import gradio as gr
from ultralytics import YOLO


# =====================================
# 1. Load YOLO11s Model
# =====================================

model = YOLO("best.pt")


# =====================================
# 2. Detection Function
# =====================================

def detect_damage(image, confidence):

    if image is None:
        return None, [], "Please upload an image first."

    # Run YOLO prediction
    results = model.predict(
        source=image,
        imgsz=832,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes
    annotated_image = result.plot()[:, :, ::-1]

    # Extract detection information
    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())

            class_name = result.names[class_id]

            conf = float(box.conf.item())

            detections.append([
                class_name,
                round(conf, 4)
            ])

    # Detection status
    count = len(detections)

    if count > 0:

        status = (
            f"Detection completed successfully! "
            f"Found {count} damaged areas."
        )

    else:

        status = "No road damage detected."

    return annotated_image, detections, status


# =====================================
# 3. Custom CSS
# =====================================

custom_css = """

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 25px !important;
}

#main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 5px;
}

#subtitle {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 30px;
}

#detect-btn {
    min-height: 48px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
}

#status {
    text-align: center;
}

.image-panel {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

"""


# =====================================
# 4. Gradio Interface
# =====================================

with gr.Blocks(
    title="Road Damage Detection"
) as demo:


    # Header
    gr.Markdown(
        """
        # 🚧 Road Damage Detection
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        AI-powered road damage detection
        
        """,
        elem_id="subtitle"
    )


    # Input and Output Images
    with gr.Row(equal_height=True):

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="Input Road Image",
                height=400
            )


        with gr.Column(scale=1):

            output_image = gr.Image(
                type="numpy",
                label="Detection Results",
                height=400
            )


    # Confidence and Detection Button
    with gr.Row(equal_height=True):

        confidence = gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=0.25,
            step=0.05,
            label="Confidence Threshold",
            scale=3
        )

        detect_button = gr.Button(
            "🔍 Detect Road Damage",
            variant="primary",
            elem_id="detect-btn",
            scale=1
        )


    # Detection Status
    status = gr.Textbox(
        label="Detection Status",
        interactive=False,
        elem_id="status"
    )


    # Results Table
    gr.Markdown(
        """
        ### 📊 Detection Summary
        """
    )

    output_table = gr.Dataframe(
        headers=[
            "Damage Class",
            "Confidence"
        ],
        datatype=[
            "str",
            "number"
        ],
        interactive=False,
        wrap=True
    )


    # Button Event
    detect_button.click(
        fn=detect_damage,
        inputs=[
            input_image,
            confidence
        ],
        outputs=[
            output_image,
            output_table,
            status
        ]
    )


# =====================================
# 5. Launch Application
# =====================================

if __name__ == "__main__":

    demo.launch(

        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        ),

        css=custom_css

    )

import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr

model = tf.saved_model.load("/Users/lama/Downloads/saved_model")
infer = model.signatures["serving_default"]

custom_css = """
/* ── page background ── */
body, .gradio-container {
    background: #f5f4f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── hide default footer ── */
footer { display: none !important; }

/* ── main card panels ── */
.panel-card {
    background: #ffffff !important;
    border: 0.5px solid #e0ddd6 !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
}

/* ── upload zone ── */
.upload-zone .wrap {
    border: 1.5px dashed #c8c6be !important;
    border-radius: 12px !important;
    background: #f5f4f0 !important;
    min-height: 200px !important;
}
.upload-zone .wrap:hover {
    border-color: #a0a09a !important;
    background: #eceae4 !important;
}

/* ── classify button ── */
.classify-btn {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
.classify-btn:hover { background: #333333 !important; }
.classify-btn:disabled { background: #c0bfbb !important; cursor: default !important; }

/* ── label output widget ── */
.label-output .label-container {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 0.5px solid #e0ddd6 !important;
    padding: 1rem !important;
}
.label-output .label-container .label {
    font-size: 20px !important;
    font-weight: 500 !important;
    color: #1a1a1a !important;
}
.label-output .bar-wrap { margin-top: 8px !important; }
.label-output .bar {
    height: 8px !important;
    border-radius: 99px !important;
}
.label-output [data-label="Cat 🐱"] .bar { background: #5DCAA5 !important; }
.label-output [data-label="Dog 🐶"] .bar { background: #AFA9EC !important; }

/* ── stat boxes ── */
.stat-box {
    background: #f5f4f0 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    text-align: center !important;
}
.stat-box .stat-label {
    font-size: 12px !important;
    color: #888780 !important;
    margin-bottom: 4px !important;
}
.stat-box .stat-value {
    font-size: 22px !important;
    font-weight: 500 !important;
    color: #1a1a1a !important;
}

/* ── history rows ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 0.5px solid #e8e6e0;
    font-size: 13px;
    color: #444;
}
.history-item:last-child { border-bottom: none; }
.history-badge-cat {
    background: #E1F5EE; color: #085041;
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 99px;
}
.history-badge-dog {
    background: #EEEDFE; color: #3C3489;
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 99px;
}

/* ── section labels ── */
.section-label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #888780 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}
"""

session_state = {
    "total": 0,
    "cats": 0,
    "dogs": 0,
    "conf_sum": 0.0,
    "history": []
}

def predict(img):
    if img is None:
        return None, "—", "—", "—", "—", "<p style='color:#888;font-size:13px;'>No classifications yet.</p>"

    pil = Image.fromarray(img).convert("RGB").resize((224, 224))
    arr = np.array(pil) / 255.0
    tensor = tf.convert_to_tensor(np.expand_dims(arr, axis=0), dtype=tf.float32)
    output = infer(tensor)
    p = float(list(output.values())[0].numpy()[0][0])

    dog_prob = round(p, 4)
    cat_prob = round(1 - p, 4)
    is_dog = p > 0.5
    conf = dog_prob if is_dog else cat_prob
    label = "Dog 🐶" if is_dog else "Cat 🐱"

    s = session_state
    s["total"] += 1
    s["conf_sum"] += conf
    if is_dog:
        s["dogs"] += 1
    else:
        s["cats"] += 1

    avg_conf = f"{round(s['conf_sum'] / s['total'] * 100)}%"
    total_str = str(s["total"])
    cats_str = str(s["cats"])
    dogs_str = str(s["dogs"])

    badge_class = "history-badge-dog" if is_dog else "history-badge-cat"
    s["history"].insert(0, {
        "label": label,
        "conf": round(conf * 100),
        "badge": badge_class
    })
    if len(s["history"]) > 5:
        s["history"].pop()

    history_html = "".join([
        f"""<div class='history-item'>
              <span style='flex:1;font-weight:500;'>{h['label']}</span>
              <span style='color:#888;'>Run #{i+1} &middot; {h['conf']}% confidence</span>
              <span class='{h['badge']}'>{h['label'].split()[0]}</span>
            </div>"""
        for i, h in enumerate(s["history"])
    ])

    return (
        {"Dog 🐶": dog_prob, "Cat 🐱": cat_prob},
        total_str,
        avg_conf,
        cats_str,
        dogs_str,
        history_html
    )

with gr.Blocks(css=custom_css, title="Cat vs Dog Classifier") as app:

    gr.HTML("""
        <div style='padding: 1.5rem 0 0.5rem;'>
            <h1 style='font-size:22px;font-weight:500;color:#1a1a1a;margin:0;'>Cat vs dog classifier</h1>
            <p style='font-size:14px;color:#888780;margin:4px 0 0;'>Upload a photo to classify it instantly</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML("<p class='section-label'>Upload image</p>")
            image_input = gr.Image(
                type="numpy",
                label="",
                elem_classes="upload-zone",
                show_label=False
            )
            classify_btn = gr.Button(
                "Classify image",
                elem_classes="classify-btn"
            )

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML("<p class='section-label'>Result</p>")
            label_output = gr.Label(
                num_top_classes=2,
                label="",
                elem_classes="label-output",
                show_label=False
            )

    with gr.Row():
        with gr.Column(elem_classes="panel-card"):
            gr.HTML("<p class='section-label'>Session stats</p>")
            with gr.Row():
                with gr.Column(elem_classes="stat-box"):
                    gr.HTML("<div class='stat-label'>Total runs</div>")
                    stat_total = gr.Text(value="0", show_label=False, container=False, interactive=False)
                with gr.Column(elem_classes="stat-box"):
                    gr.HTML("<div class='stat-label'>Avg confidence</div>")
                    stat_avg = gr.Text(value="—", show_label=False, container=False, interactive=False)
            with gr.Row():
                with gr.Column(elem_classes="stat-box"):
                    gr.HTML("<div class='stat-label'>🐱 Cats</div>")
                    stat_cats = gr.Text(value="0", show_label=False, container=False, interactive=False)
                with gr.Column(elem_classes="stat-box"):
                    gr.HTML("<div class='stat-label'>🐶 Dogs</div>")
                    stat_dogs = gr.Text(value="0", show_label=False, container=False, interactive=False)

    with gr.Row(elem_classes="panel-card"):
        with gr.Column():
            gr.HTML("<p class='section-label'>Recent classifications</p>")
            history_html = gr.HTML("<p style='color:#888;font-size:13px;'>No classifications yet.</p>")

    classify_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[label_output, stat_total, stat_avg, stat_cats, stat_dogs, history_html]
    )

app.launch(share=True)
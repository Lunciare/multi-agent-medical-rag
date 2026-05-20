import gradio as gr
from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR

try:
    print("Initializing orchestrator...")
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    print("Orchestrator ready!")
except Exception:
    import traceback; traceback.print_exc()
    raise SystemExit("Orchestrator failed to initialise; aborting.")

def process_query(query):
    specialist, response, evidence = orchestrator.answer(query)
    return specialist.title(), response, evidence

with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Multi-Agent Medical RAG System")
    gr.Markdown("Clinical Decision Support Assistant for Cardiology & Endocrinology.")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                lines=4,
                label="Clinical Query",
                placeholder="Enter a clinical scenario (e.g., A 58-year-old female presents with episodic palpitations...)"
            )
            submit_btn = gr.Button("Analyze Query", variant="primary")

        with gr.Column(scale=1):
            specialist_badge = gr.Textbox(
                label="Routed Specialist",
                interactive=False,
                placeholder="Waiting for routing..."
            )

    with gr.Row():
        response_output = gr.Markdown(label="Structured Response")

    with gr.Row():
        with gr.Accordion("Retrieved Chunks & Confidence (FAISS)", open=False):
            evidence_output = gr.Markdown()

    submit_btn.click(
        fn=process_query,
        inputs=[query_input],
        outputs=[specialist_badge, response_output, evidence_output]
    )

    gr.Examples(
        examples=[
            ["A 58-year-old female presents with episodic palpitations and shortness of breath on exertion that have worsened over the past three weeks. She has no prior cardiac history. What are the potential cardiac causes, and what initial workup is recommended?"],
            ["A 65-year-old male with a 20-year history of poorly controlled hypertension reports substernal chest pressure radiating to the left arm during physical activity, relieved by rest. How do these chest pain patterns correlate with his hypertension as a risk factor, and what differential diagnoses should be considered?"],
            ["A 45-year-old patient shares data from a consumer wearable showing intermittent irregular R-R intervals and reduced heart rate variability over the past month. What cardiac conditions could explain these findings, and when should a formal 12-lead ECG or Holter monitor be ordered?"],
            ["A 52-year-old male experiences recurrent episodes of dizziness and a sensation of a racing heartbeat lasting 10-15 minutes. An initial resting ECG was normal. What diagnostic procedures are indicated for suspected paroxysmal arrhythmia, and in what order should they be pursued?"],
            ["A 60-year-old patient with a BMI of 32, moderate alcohol intake, and untreated sleep apnea has been found to have occasional atrial fibrillation on a recent Holter recording. What are the key modifiable risk factors contributing to atrial fibrillation in this patient, and what lifestyle interventions does the evidence support?"],
            ["A 42-year-old female presents with progressive fatigue, unexplained weight gain of 8 kg over 6 months, cold intolerance, and dry skin. TSH is 12.4 mIU/L (reference 0.4–4.0) and free T4 is 0.6 ng/dL (reference 0.8–1.8). What is the likely diagnosis, what additional workup is appropriate, and what are the key considerations for management?"],
            ["A 55-year-old male with BMI 34, waist circumference 110 cm, fasting glucose 148 mg/dL, and HbA1c 7.8% is diagnosed with type 2 diabetes. He also has hypertension (BP 145/92) and dyslipidemia (triglycerides 280 mg/dL, HDL 32 mg/dL). What are the key components of initial assessment, and what evidence-based approach should be taken for glycemic and cardiovascular risk management?"],
            ["A 50-year-old female undergoes a CT scan for abdominal pain, which incidentally reveals a 2.5 cm left adrenal mass with Hounsfield units of 8. She has no symptoms of hormone excess. What is the recommended workup for an adrenal incidentaloma, what biochemical tests should be performed to rule out functional tumors, and what are the follow-up guidelines?"],
            ["A 62-year-old postmenopausal female is found to have a serum calcium of 11.2 mg/dL (reference 8.5–10.5) and an intact PTH of 98 pg/mL (reference 15–65). She reports fatigue, polyuria, and mild constipation. What is the differential diagnosis for PTH-dependent hypercalcemia, what additional investigations are indicated, and what are the criteria for surgical intervention?"],
            ["A 38-year-old male undergoes an MRI for chronic headaches, which reveals a 12 mm pituitary macroadenoma. He reports no visual disturbances. What hormonal evaluation should be performed to assess for hypersecretion and hypopituitarism, and what are the indications for surgical versus medical management?"]
        ],
        inputs=query_input,
        label="Select a test case to run:"
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())

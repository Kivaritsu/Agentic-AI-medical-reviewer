# src/medical/app.py
# Streamlit front-end: accepts Excel/PDF (and other files), saves them to uploads/,
# previews them, and passes the saved local path into the MedicalAnalysisCrew.

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Attempt to import crew
try:
    from medical.crew import get_medical_crew
except Exception as e:
    st.error("❌ Failed to import MedicalAnalysisCrew.\n"
             "Make sure you started Streamlit from the project ROOT:\n"
             "`python -m streamlit run src/medical/app.py`")
    st.exception(e)
    st.stop()

# Load environment
try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[2]
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except:
    pass

st.set_page_config(page_title="Medical AI Suite", layout="wide")

# -----------------------------------------------------------
# Sidebar – API keys
# -----------------------------------------------------------
st.sidebar.title("🔐 API Status")

if os.environ.get("GEMINI_API_KEY"):
    st.sidebar.success("Gemini API Key Loaded ✓")
else:
    st.sidebar.error("GEMINI_API_KEY missing ✘")

# -----------------------------------------------------------
# Prepare uploads folder
# -----------------------------------------------------------
PROJECT_ROOT = Path.cwd()
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📄 Medical Analysis", "🤖 ML Prediction", "✨ Gemini Text Generation"])

# Lazy crew instance
@st.cache_resource
def _load_crew():
    return get_medical_crew()

crew = _load_crew()

# -------------------------
# Helper functions for previewing files
# -------------------------
def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded_file (streamlit UploadedFile) to uploads/ and return the absolute path as string.
    """
    dest = UPLOADS_DIR / uploaded_file.name
    # avoid overwrite collision by adding suffix if exists
    i = 1
    base = dest.stem
    suf = dest.suffix
    while dest.exists():
        dest = UPLOADS_DIR / f"{base}_{i}{suf}"
        i += 1
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest.resolve())

def preview_spreadsheet(path: str, nrows: int = 10):
    import pandas as pd
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path, nrows=nrows)
        else:
            df = pd.read_excel(path, nrows=nrows)
        st.write(df)
    except Exception as e:
        st.warning(f"Couldn't preview spreadsheet: {e}")

def preview_pdf(path: str):
    # Attempt a lightweight PDF text extract using PyPDF2 if available
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            first_page_text = ""
            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text() or ""
        if first_page_text.strip():
            st.text_area("First page text (preview)", value=first_page_text, height=300)
        else:
            st.info("PDF preview produced no extractable text (may be scanned image).")
    except Exception:
        st.info("Install PyPDF2 to enable PDF previews (optional).")

def preview_text(path: str):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        st.text_area("File preview (first 5000 chars)", value=text[:5000], height=300)
    except Exception as e:
        st.warning(f"Couldn't preview file: {e}")

# -----------------------------------------------------------
# TAB 1 — Medical Analysis using CrewAI Tasks
# -----------------------------------------------------------
with tab1:
    st.header("📄 Full Medical Document Analysis")
    st.caption("Upload a medical document (Excel / CSV / PDF / TXT / DOCX) and run the CrewAI pipeline.")

    patient_id = st.text_input("Patient ID", value="demo_patient")

    uploaded = st.file_uploader("Upload a file (Excel/CSV/PDF/TXT/DOCX)", type=["pdf", "csv", "xls", "xlsx", "txt", "docx"])
    saved_path = None
    if uploaded:
        saved_path = save_uploaded_file(uploaded)
        st.success(f"Saved upload to: `{saved_path}`")
        # Show preview depending on type
        if saved_path.lower().endswith((".xls", ".xlsx", ".csv")):
            preview_spreadsheet(saved_path)
        elif saved_path.lower().endswith(".pdf"):
            preview_pdf(saved_path)
        else:
            preview_text(saved_path)

    # Or allow manual path input (for advanced users)
    demo_file = "/mnt/data/app.py"  # example per instructions
    manual_path = st.text_input("Or provide existing local file path (optional)", value=demo_file)
    # prefer uploaded path if present
    medical_path = saved_path or (manual_path.strip() if manual_path.strip() else None)

    if st.button("Run Full Analysis"):
        if not medical_path:
            st.warning("Please upload a file or supply a valid path.")
        else:
            st.info("⏳ Running CrewAI medical analysis...")
            try:
                # transform local path to crew-friendly 'url' (crew knows how to handle the historic path too)
                file_url = crew.file_path_as_url(medical_path)
                # pass file url into run_analysis (crew.run_analysis expects medical_records_path)
                output = crew.run_analysis(patient_id=patient_id, medical_records_path=file_url)
                st.success("Analysis Complete ✓")
                st.json(output)
            except Exception as e:
                st.error("❌ Error running analysis")
                st.exception(e)

# -----------------------------------------------------------
# TAB 2 — ML Prediction
# -----------------------------------------------------------
with tab2:
    st.header("🤖 ML Model Prediction")
    st.caption("Use your trained model to predict from a clinical note or an uploaded file.")

    clinical_note = st.text_area("Enter patient clinical note:", height=220)

    uploaded_ml = st.file_uploader("Or upload a CSV/XLSX with clinical notes (ML):", type=["csv", "xls", "xlsx"], key="mlfile")
    ml_saved_path = None
    if uploaded_ml:
        ml_saved_path = save_uploaded_file(uploaded_ml)
        st.success(f"Saved ML upload to: `{ml_saved_path}`")
        st.caption("This will run prediction on the first textual column of the sheet (first 10 rows preview).")
        try:
            import pandas as pd
            df = pd.read_csv(ml_saved_path) if ml_saved_path.lower().endswith(".csv") else pd.read_excel(ml_saved_path)
            st.write(df.head(10))
        except Exception as e:
            st.warning(f"Preview failed: {e}")

    if st.button("Predict from ML Model"):
        try:
            ml_tool = crew.ml_prediction_tool()
            if clinical_note and clinical_note.strip():
                result = ml_tool.predict_text(clinical_note)
            elif ml_saved_path:
                # read first textual column from file and run prediction on each row (limited to 10 rows)
                import pandas as pd
                df = pd.read_csv(ml_saved_path) if ml_saved_path.lower().endswith(".csv") else pd.read_excel(ml_saved_path)
                # heuristics: pick first column with dtype object
                text_cols = [c for c in df.columns if df[c].dtype == object]
                if not text_cols:
                    st.warning("No textual column detected in uploaded file.")
                    st.stop()
                col = text_cols[0]
                preview_texts = df[col].dropna().astype(str).head(10).tolist()
                results = [ml_tool.predict_text(t) for t in preview_texts]
                result = {"batch_preview_count": len(results), "results": results}
            else:
                st.warning("Enter a clinical note or upload a file.")
                st.stop()
            st.success("Prediction Ready ✓")
            st.json(result)
        except Exception as e:
            st.error("❌ ML Prediction Failed")
            st.exception(e)

# -----------------------------------------------------------
# TAB 3 — Gemini Free-Text Generation
# -----------------------------------------------------------
with tab3:
    st.header("✨ Gemini Text Generation")
    st.caption("Send any medical text or note to Gemini and get summarized output.")

    user_prompt = st.text_area("Enter prompt or medical text:", height=220)

    use_file = st.checkbox("Attach uploaded file as URL?", value=False)
    gem_file_url = None
    if use_file:
        # allow selecting a file saved in uploads/ (or type a path)
        files = sorted([str(p.name) for p in UPLOADS_DIR.iterdir()]) if UPLOADS_DIR.exists() else []
        selected = st.selectbox("Choose saved upload to attach (optional)", options=[""] + files)
        if selected:
            gem_file_path = str((UPLOADS_DIR / selected).resolve())
            gem_file_url = crew.file_path_as_url(gem_file_path)

    model_name = st.text_input("Model (optional):", value=os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-pro"))
    max_tokens = st.slider("Max tokens", 50, 2000, 512)

    if st.button("Generate with Gemini"):
        if not user_prompt.strip():
            st.warning("Enter prompt text first.")
        else:
            with st.spinner("Contacting Gemini..."):
                try:
                    gem = crew.gemini_tool()
                    prompt = user_prompt
                    if gem_file_url:
                        prompt += f"\n\nAttached file URL: {gem_file_url}"
                    response = gem.generate(prompt, model=model_name, params={"max_tokens": max_tokens})
                    st.success("Generation Complete ✓")
                    st.subheader("Generated Text")
                    st.code(response.get("text", ""))
                    st.subheader("Raw Response")
                    st.json(response.get("raw"))
                except Exception as e:
                    st.error("❌ Gemini generation failed")
                    st.exception(e)

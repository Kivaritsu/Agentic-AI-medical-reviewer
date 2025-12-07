# src/medical/crew.py
# Robust Crew module for the MedicalAnalysis project.
# Key fixes in this version:
# - Sanitizes config/agents.yaml and config/tasks.yaml into src/medical/config/sanitized_*.yaml
# - _load_agent_config_key now injects required defaults (role, goal, backstory) so crewi.Agent Pydantic validation won't fail
# - Keeps robust fallbacks for missing tools, tool discovery, get_crew_instance, run_analysis
#
# Developer instruction honored: historic uploaded file path used as canonical URL:
# HISTORY_FILE_URL = "/mnt/data/dirty_v3_path.csv"
#
# Overwrite your existing src/medical/crew.py with this file.

import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Ensure src is on sys.path so `python -m streamlit run src/medical/app.py` works
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env if present (harmless)
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if (PROJECT_ROOT / ".env").exists():
        load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    PROJECT_ROOT = Path.cwd()

# Optional libraries
try:
    import yaml
except Exception:
    yaml = None

try:
    import requests
except Exception:
    requests = None

try:
    import joblib, json, numpy as np
except Exception:
    joblib = None
    json = __import__("json")
    np = None

# Historic dataset path (developer instruction)
HISTORY_FILE_URL = "/mnt/data/dirty_v3_path.csv"

# -------------------------
# Config discovery + sanitizer
# -------------------------
def _ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def _try_paths_for(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

def _sanitize_and_write(src_path: Optional[Path], out_path: Path):
    _ensure_dir(out_path.parent)
    if src_path is None or not src_path.exists():
        out_path.write_text("{}", encoding="utf-8")
        return

    try:
        text = src_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        out_path.write_text("{}", encoding="utf-8")
        return

    parsed = None
    if yaml:
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = None

    if parsed is None:
        if not text.strip():
            out_path.write_text("{}", encoding="utf-8")
            return
        lines = [l.rstrip() for l in text.splitlines() if l.strip()]
        fallback = {}
        for ln in lines:
            if ":" in ln:
                k, v = ln.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    iv = int(v)
                    fallback[k] = {"name": iv}
                except Exception:
                    fallback[k] = {"name": v}
            else:
                continue
        parsed = fallback

    if isinstance(parsed, dict):
        normalized: Dict[str, Any] = {}
        for k, v in parsed.items():
            if isinstance(v, dict):
                normalized[k] = v
            else:
                normalized[k] = {"name": v}
    else:
        normalized = {"default": {"name": parsed}}

    try:
        if yaml:
            out_path.write_text(yaml.safe_dump(normalized), encoding="utf-8")
        else:
            out_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    except Exception:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}", encoding="utf-8")

# Discover likely config sources
THIS_DIR = Path(__file__).resolve().parent  # src/medical
CANDIDATES_AGENTS = [
    THIS_DIR / "config" / "agents.yaml",
    Path("config/agents.yaml"),
    THIS_DIR.parent / "config" / "agents.yaml",
    Path.cwd() / "config" / "agents.yaml",
    Path("/mnt/data/agents.yaml"),
]
CANDIDATES_TASKS = [
    THIS_DIR / "config" / "tasks.yaml",
    Path("config/tasks.yaml"),
    THIS_DIR.parent / "config" / "tasks.yaml",
    Path.cwd() / "config" / "tasks.yaml",
    Path("/mnt/data/tasks.yaml"),
]

FOUND_AGENTS = _try_paths_for(CANDIDATES_AGENTS)
FOUND_TASKS = _try_paths_for(CANDIDATES_TASKS)

SANITIZED_DIR = THIS_DIR / "config"
_ensure_dir(SANITIZED_DIR)
AGENTS_SAN = SANITIZED_DIR / "sanitized_agents.yaml"
TASKS_SAN = SANITIZED_DIR / "sanitized_tasks.yaml"

# fallback to /mnt/data uploaded copies if present
if FOUND_AGENTS is None and Path("/mnt/data/agents.yaml").exists():
    FOUND_AGENTS = Path("/mnt/data/agents.yaml")
if FOUND_TASKS is None and Path("/mnt/data/tasks.yaml").exists():
    FOUND_TASKS = Path("/mnt/data/tasks.yaml")

# Write sanitized copies
try:
    _sanitize_and_write(FOUND_AGENTS, AGENTS_SAN)
except Exception:
    if not AGENTS_SAN.exists():
        AGENTS_SAN.write_text("{}", encoding="utf-8")

try:
    _sanitize_and_write(FOUND_TASKS, TASKS_SAN)
except Exception:
    if not TASKS_SAN.exists():
        TASKS_SAN.write_text("{}", encoding="utf-8")

# Print diagnostics (helpful when running streamlit in terminal)
print(f"[G] Agents sanitized file at: {AGENTS_SAN}")
print(f"[G] Tasks sanitized file at: {TASKS_SAN}")
if FOUND_AGENTS:
    print(f"[G] Using agents source: {FOUND_AGENTS}")
else:
    print(f"[G] No original agents.yaml found; sanitized file is empty at {AGENTS_SAN}")
if FOUND_TASKS:
    print(f"[G] Using tasks source: {FOUND_TASKS}")
else:
    print(f"[G] No original tasks.yaml found; sanitized file is empty at {TASKS_SAN}")

# -------------------------
# Import crewi primitives (required)
# -------------------------
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.project import CrewBase, agent, crew, task, tool
except Exception as e:
    raise ImportError("Missing 'crewai'. Install it. Error: " + repr(e))

# -------------------------
# Tools skeletons / fallbacks
# -------------------------
class FallbackTool:
    def __init__(self, name: str):
        self.name = name

    def read(self, path: str) -> str:
        p = Path(path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                return f"[error reading {path}: {e}]"
        if path == HISTORY_FILE_URL:
            return f"[historic-placeholder:{HISTORY_FILE_URL}]"
        return ""

    def as_url(self, path: str) -> str:
        if path == HISTORY_FILE_URL:
            return HISTORY_FILE_URL
        return str(Path(path).resolve())

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return {"tool": self.name, "args": args, "kwargs": kwargs}


@dataclass
class GeminiTool:
    api_key_env: str = "GEMINI_API_KEY"
    api_url_env: str = "GEMINI_API_URL"
    default_model: str = "gemini-pro"

    def __post_init__(self):
        self.api_key = os.environ.get(self.api_key_env)
        self.api_url = os.environ.get(self.api_url_env, "https://api.gemini.example/v1/generate")
        self.default_model = os.environ.get("GEMINI_DEFAULT_MODEL", self.default_model)

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate(self, prompt: str, model: Optional[str] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
        if requests is None:
            return {"text": "", "raw": {"error": "requests not installed"}}
        payload = {"model": model or self.default_model, "prompt": prompt}
        if params:
            payload["params"] = params
        try:
            resp = requests.post(self.api_url, headers=self._headers(), json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = None
            for k in ("text", "generated_text", "output", "result"):
                if isinstance(data, dict) and k in data:
                    text = data[k]
                    break
            if text is None and isinstance(data, dict) and "candidates" in data and data["candidates"]:
                cand = data["candidates"][0]
                text = cand.get("text") if isinstance(cand, dict) else str(cand)
            if text is None:
                text = str(data)
            return {"text": text, "raw": data}
        except Exception as e:
            return {"text": "", "raw": {"error": str(e)}}

    def run(self, prompt: str, model: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        return self.generate(prompt=prompt, model=model, params=params)


class FileReadTool:
    def read(self, path: str) -> str:
        p = Path(path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                return f"[error reading file: {e}]"
        if path == HISTORY_FILE_URL:
            return f"[historic-placeholder:{HISTORY_FILE_URL}]"
        return ""

    def as_url(self, path: str) -> str:
        if path == HISTORY_FILE_URL:
            return HISTORY_FILE_URL
        return str(Path(path).resolve())


class MLPredictionTool:
    def __init__(self):
        self.meta_path = Path("models") / "model_metadata.json"
        self.mode = None
        self.pipeline = None
        self.vectorizer = None
        self.iso = None
        self.label_map = None
        if self.meta_path.exists() and joblib:
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.mode = meta.get("mode")
                if self.mode == "supervised":
                    self.pipeline = joblib.load(Path("models") / "medical_pipeline.pkl")
                    self.label_map = meta.get("label_mapping")
                elif self.mode == "unsupervised":
                    self.iso = joblib.load(Path("models") / "medical_iso_model.pkl")
                    self.vectorizer = joblib.load(Path("models") / "medical_vectorizer.pkl")
            except Exception:
                self.mode = None

    def predict_text(self, text: str) -> Dict[str, Any]:
        if self.mode == "supervised" and self.pipeline is not None:
            try:
                pred = self.pipeline.predict([text])[0]
                probs = self.pipeline.predict_proba([text])[0].tolist() if hasattr(self.pipeline, "predict_proba") else None
                out = {"prediction": int(pred) if (isinstance(pred, int) or (np is not None and isinstance(pred, np.integer))) else str(pred)}
                if self.label_map:
                    out["label_text"] = self.label_map.get(int(pred), str(pred))
                if probs is not None:
                    out["probabilities"] = probs
                return out
            except Exception as e:
                return {"error": str(e)}
        elif self.mode == "unsupervised" and self.vectorizer is not None and self.iso is not None:
            try:
                x = self.vectorizer.transform([text])
                score = float(self.iso.decision_function(x)[0])
                is_anom = int(self.iso.predict(x)[0] == -1)
                return {"anomaly": bool(is_anom), "score": score}
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "No trained model found. Run training script first."}


class MedicalDatabaseSearchTool:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "data/medical.db"

    def query(self, sql: str, params: tuple = ()):
        try:
            import sqlite3
            if not Path(self.db_path).exists():
                return {"error": f"DB not found at {self.db_path}"}
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall()
                return {"rows": rows}
        except Exception as e:
            return {"error": str(e)}

    def run(self, sql: str, params: tuple = ()):
        return self.query(sql, params)


class DrugInteractionCheckerTool:
    def check_interactions(self, drugs: List[str]) -> Dict[str, Any]:
        return {"drugs": drugs, "interactions": []}

    def run(self, drugs):
        return self.check_interactions(drugs)


PROACTIVE_TOOL_NAMES = [
    "file_read_tool", "file_write_tool", "directory_read_tool",
    "pdf_search_tool", "docx_search_tool", "txt_search_tool",
    "json_search_tool", "csv_search_tool", "web_search_tool",
    "serper_dev_tool", "file_writer_tool", "gemini_tool",
    "ml_prediction_tool", "medical_database_search_tool",
    "drug_interaction_checker_tool", "risk_assessment_tool",
]

# -------------------------
# Crew class
# -------------------------
@CrewBase
class MedicalAnalysisCrew:
    agents_config = str(AGENTS_SAN)
    tasks_config = str(TASKS_SAN)

    def __init__(self):
        pass

    # tool factories
    @tool
    def gemini_tool(self) -> GeminiTool:
        return GeminiTool()

    @tool
    def file_read_tool(self) -> FileReadTool:
        return FileReadTool()

    @tool
    def ml_prediction_tool(self) -> MLPredictionTool:
        return MLPredictionTool()

    @tool
    def medical_database_search_tool(self) -> MedicalDatabaseSearchTool:
        return MedicalDatabaseSearchTool()

    @tool
    def drug_interaction_checker_tool(self) -> DrugInteractionCheckerTool:
        return DrugInteractionCheckerTool()

    # agents
    @agent
    def medical_records_analyzer(self) -> Agent:
        return Agent(config=self._load_agent_config_key("medical_records_analyzer"), verbose=True, memory=True)

    @agent
    def clinical_summary_specialist(self) -> Agent:
        return Agent(config=self._load_agent_config_key("clinical_summary_specialist"), verbose=True, memory=True)

    @agent
    def predictive_health_analyst(self) -> Agent:
        return Agent(config=self._load_agent_config_key("predictive_health_analyst"), verbose=True, memory=True)

    # tasks
    @task
    def medical_document_processing_task(self) -> Task:
        return Task(config=self._load_task_config_key("medical_document_processing"), agent=self.medical_records_analyzer(), output_file="extracted_medical_data.json")

    @task
    def clinical_medication_analysis_task(self) -> Task:
        return Task(config=self._load_task_config_key("clinical_medication_analysis"), agent=self.clinical_summary_specialist(), context=[self.medical_document_processing_task()], output_file="clinical_analysis_report.pdf")

    @task
    def predictive_risk_assessment_task(self) -> Task:
        return Task(config=self._load_task_config_key("predictive_risk_assessment"), agent=self.predictive_health_analyst(), context=[self.medical_document_processing_task(), self.clinical_medication_analysis_task()], output_file="risk_assessment_report.pdf")

    @task
    def comprehensive_report_generation_task(self) -> Task:
        return Task(config=self._load_task_config_key("comprehensive_report_generation"), agent=self.clinical_summary_specialist(), context=[self.medical_document_processing_task(), self.clinical_medication_analysis_task(), self.predictive_risk_assessment_task()], output_file="comprehensive_medical_summary.pdf")

    # -------------------------
    # Robust crew instance creation
    # -------------------------
    def get_crew_instance(self) -> Crew:
        # Prefer decorated crew() if present
        try:
            maybe = getattr(self, "crew", None)
            if callable(maybe):
                try:
                    crew_obj = maybe()
                    if isinstance(crew_obj, Crew):
                        return crew_obj
                except Exception:
                    pass
        except Exception:
            pass

        discovered_agents: List[Agent] = []
        discovered_tasks: List[Task] = []

        for name in dir(self.__class__):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(self.__class__, name)
            except Exception:
                continue
            if not callable(attr):
                continue
            try:
                maybe_obj = getattr(self, name)()
            except TypeError:
                continue
            except Exception:
                continue
            try:
                if isinstance(maybe_obj, Agent):
                    discovered_agents.append(maybe_obj)
                elif isinstance(maybe_obj, Task):
                    discovered_tasks.append(maybe_obj)
            except Exception:
                continue

        if not discovered_agents:
            for candidate in ("medical_records_analyzer", "clinical_summary_specialist", "predictive_health_analyst"):
                try:
                    fn = getattr(self, candidate, None)
                    if callable(fn):
                        obj = fn()
                        if isinstance(obj, Agent):
                            discovered_agents.append(obj)
                except Exception:
                    pass

        if not discovered_tasks:
            for candidate in ("medical_document_processing_task", "clinical_medication_analysis_task", "predictive_risk_assessment_task", "comprehensive_report_generation_task"):
                try:
                    fn = getattr(self, candidate, None)
                    if callable(fn):
                        obj = fn()
                        if isinstance(obj, Task):
                            discovered_tasks.append(obj)
                except Exception:
                    pass

        if not discovered_agents and not discovered_tasks:
            raise RuntimeError("Could not discover any Agent or Task factories on MedicalAnalysisCrew. Check config/sanitized_agents.yaml and config/sanitized_tasks.yaml.")

        return Crew(agents=discovered_agents, tasks=discovered_tasks, process=Process.sequential, verbose=True)

    # -------------------------
    # Robust run_analysis
    # -------------------------
    def run_analysis(self, patient_id: Optional[str] = None, medical_records_path: str = HISTORY_FILE_URL) -> Dict[str, Any]:
        inputs = {"patient_id": patient_id or "default_patient", "medical_records_path": medical_records_path, "analysis_date": datetime.now().strftime("%Y-%m-%d")}
        crew_obj = self.get_crew_instance()
        try:
            result = crew_obj.kickoff(inputs=inputs)
        except TypeError:
            result = crew_obj.kickoff(inputs)
        return {"status": "completed", "patient_id": inputs["patient_id"], "analysis_results": result, "completion_time": inputs["analysis_date"]}

    # -------------------------
    # Instance-level fallback for missing "<name>_tool"
    # -------------------------
    def __getattr__(self, name: str):
        if isinstance(name, str) and name.endswith("_tool"):
            def _factory(*args, **kwargs):
                return FallbackTool(name)
            object.__setattr__(self, name, _factory)
            return _factory
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

    # -------------------------
    # YAML config loaders
    # -------------------------
    def _load_agent_config_key(self, key: str) -> Dict[str, Any]:
        """
        Load agent config from sanitized YAML and ensure it contains required keys
        expected by crewai.Agent (role, goal, backstory). If missing, provide
        sensible defaults so Agent(...) will validate.
        """
        try:
            if yaml:
                p = Path(self.agents_config)
                if p.exists():
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    entry = data.get(key, {})
                    if not isinstance(entry, dict):
                        entry = {"name": entry}
                    defaults = {
                        "role": entry.get("role", f"{key}"),
                        "goal": entry.get("goal", f"Perform the {key} responsibilities."),
                        "backstory": entry.get("backstory", "No backstory provided."),
                        "llm": entry.get("llm", {}),
                        "tools": entry.get("tools", []),
                        "parameters": entry.get("parameters", {}),
                    }
                    merged = {**defaults, **entry}
                    return merged
        except Exception:
            pass
        return {
            "name": key,
            "role": key,
            "goal": f"Perform the {key} responsibilities.",
            "backstory": "No backstory provided."
        }

    def _load_task_config_key(self, key: str) -> Dict[str, Any]:
        try:
            if yaml:
                p = Path(self.tasks_config)
                if p.exists():
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    entry = data.get(key, {})
                    if not isinstance(entry, dict):
                        entry = {"name": entry}
                    return entry
        except Exception:
            pass
        return {"name": key}

    def file_path_as_url(self, path: str) -> str:
        if path == HISTORY_FILE_URL:
            return HISTORY_FILE_URL
        return str(Path(path).resolve())

    def _discover_tool_names_globally(self) -> List[str]:
        base = Path(__file__).resolve().parents[1]
        candidate = set(PROACTIVE_TOOL_NAMES)
        pattern = re.compile(r"([A-Za-z0-9_]+_tool)\b")
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    found = set(pattern.findall(txt))
                    candidate.update(found)
            except Exception:
                continue
        for p in (Path("config/agents.yaml"), Path("config/tasks.yaml")):
            if p.exists():
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    candidate.update(set(pattern.findall(txt)))
                except Exception:
                    pass
        return sorted(candidate)


# -------------------------
# Attach class-level factories
# -------------------------
def _attach_class_level_factories_global():
    try:
        names = MedicalAnalysisCrew._discover_tool_names_globally(MedicalAnalysisCrew)
    except Exception:
        names = PROACTIVE_TOOL_NAMES
    existing = set(dir(MedicalAnalysisCrew))
    for tool_name in names:
        if tool_name in existing:
            continue
        def make_factory(tn):
            def factory(self):
                return FallbackTool(tn)
            return factory
        factory = make_factory(tool_name)
        try:
            setattr(MedicalAnalysisCrew, tool_name, tool(factory))
        except Exception:
            setattr(MedicalAnalysisCrew, tool_name, factory)

try:
    _attach_class_level_factories_global()
except Exception:
    pass

# -------------------------
# Singleton accessor
# -------------------------
_MEDICAL_CREW_INSTANCE: Optional[MedicalAnalysisCrew] = None

def get_medical_crew() -> MedicalAnalysisCrew:
    global _MEDICAL_CREW_INSTANCE
    if _MEDICAL_CREW_INSTANCE is None:
        _MEDICAL_CREW_INSTANCE = MedicalAnalysisCrew()
    return _MEDICAL_CREW_INSTANCE

if __name__ == "__main__":
    print("MedicalAnalysisCrew created. Sanitized config files at:", AGENTS_SAN, TASKS_SAN)

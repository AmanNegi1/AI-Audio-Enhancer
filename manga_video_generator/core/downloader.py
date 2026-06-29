import os
import streamlit as st
import threading
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

# Helper to check if running inside Streamlit
def is_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

class StreamlitHubProgress(tqdm):
    _lock = threading.Lock()
    _placeholders = {}
    _bars = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id(self)
        
        if is_in_streamlit():
            with StreamlitHubProgress._lock:
                self.placeholder = st.empty()
                StreamlitHubProgress._placeholders[self.id] = self.placeholder
                if self.total:
                    # Initialize progress bar
                    self.bar = self.placeholder.progress(0.0, text=f"📥 {self.desc}: 0%")
                else:
                    self.bar = self.placeholder.text(f"📥 {self.desc}...")
                StreamlitHubProgress._bars[self.id] = self.bar

    def update(self, n=1):
        super().update(n)
        if is_in_streamlit():
            with StreamlitHubProgress._lock:
                if self.id not in StreamlitHubProgress._bars:
                    return
                desc = self.desc if self.desc else "Downloading"
                if len(desc) > 50:
                    desc = "..." + desc[-47:]
                
                if self.total:
                    percentage = min(1.0, max(0.0, self.n / self.total))
                    percent_str = f"{percentage * 100:.1f}%"
                    
                    if self.total > 1024 * 1024:
                        mb_loaded = self.n / (1024 * 1024)
                        mb_total = self.total / (1024 * 1024)
                        text = f"📥 {desc}: {percent_str} ({mb_loaded:.1f}MB / {mb_total:.1f}MB)"
                    else:
                        text = f"📥 {desc}: {percent_str} ({self.n} / {self.total})"
                    
                    StreamlitHubProgress._bars[self.id].progress(percentage, text=text)
                else:
                    mb_loaded = self.n / (1024 * 1024)
                    if mb_loaded > 0.1:
                        text = f"📥 {desc}: {mb_loaded:.1f}MB loaded"
                    else:
                        text = f"📥 {desc}: {self.n} units"
                    StreamlitHubProgress._placeholders[self.id].text(text)

    def close(self):
        super().close()
        if is_in_streamlit():
            with StreamlitHubProgress._lock:
                if self.id in StreamlitHubProgress._placeholders:
                    StreamlitHubProgress._placeholders[self.id].empty()
                    del StreamlitHubProgress._placeholders[self.id]
                if self.id in StreamlitHubProgress._bars:
                    del StreamlitHubProgress._bars[self.id]

def check_and_download_model(model_id: str, variant: str = None, ignore_patterns = None, allow_patterns = None):
    """
    Downloads a Hugging Face model using snapshot_download with Streamlit progress percentage.
    If the model is already fully cached, it returns immediately.
    """
    # Configure environment cache paths just in case
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "D:\\.cache\\huggingface"
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = "D:\\.cache"

    kwargs = {
        "repo_id": model_id,
        "resume_download": True,
    }
    
    if variant:
        # Prioritize files matching variant pattern
        kwargs["allow_patterns"] = [f"*{variant}*", "*.json", "*.txt", "*.model", "*.bin", "*.py", "*.onnx"]
    elif allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
        
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    # Check if running in Streamlit to show the progress UI
    if is_in_streamlit():
        kwargs["tqdm_class"] = StreamlitHubProgress
        st.info(f"🔍 Checking cache / downloading `{model_id}`...")
        
    # Run the download (or cache check) with automatic healing for broken cache pointers
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:
        if "No such file or directory" in str(exc) or "blobs" in str(exc) or "Errno 2" in str(exc):
            if is_in_streamlit():
                st.warning(f"⚠️ Detected incomplete cache pointer from previous interrupted run. Healing cache for `{model_id}`...")
            kwargs["force_download"] = True
            path = snapshot_download(**kwargs)
        else:
            raise exc
    return path

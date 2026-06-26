import os
import streamlit as st
import time
# Load OpenAI key rotation pool from .env or fallback to empty defaults
OPENAI_KEYS = []
openai_env = os.getenv("OPENAI_KEYS_POOL")
if openai_env:
    for item in openai_env.split(","):
        if ":" in item:
            k, owner = item.split(":", 1)
            OPENAI_KEYS.append({"key": k.strip(), "owner": owner.strip()})


class OpenAIKeyManager:
    @staticmethod
    def initialize():
        """Initializes the session state variables for OpenAI key rotation."""
        if "openai_keys_status" not in st.session_state:
            st.session_state["openai_keys_status"] = [
                {"key": k["key"], "owner": k["owner"], "status": "Active"} for k in OPENAI_KEYS
            ]
        if "openai_current_key_index" not in st.session_state:
            st.session_state["openai_current_key_index"] = 0
        if "custom_openai_key" not in st.session_state:
            st.session_state["custom_openai_key"] = ""
        if "custom_openai_key_status" not in st.session_state:
            st.session_state["custom_openai_key_status"] = "Active"

    @staticmethod
    def set_custom_key(custom_key):
        """Sets the custom user-provided OpenAI key from the UI."""
        is_streamlit = False
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except:
            pass

        if not is_streamlit:
            return

        OpenAIKeyManager.initialize()
        custom_key_cleaned = custom_key.strip() if custom_key else ""
        
        if st.session_state.get("custom_openai_key", "") != custom_key_cleaned:
            st.session_state["custom_openai_key"] = custom_key_cleaned
            st.session_state["custom_openai_key_status"] = "Active"

    @staticmethod
    def get_active_key(passed_key=None):
        """Returns the current active OpenAI key."""
        is_streamlit = False
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except:
            pass

        if not is_streamlit:
            return passed_key if passed_key else os.environ.get("OPENAI_API_KEY", OPENAI_KEYS[0]["key"])

        OpenAIKeyManager.initialize()

        # Register passed_key as custom key if provided and not in rotation pool
        if passed_key and passed_key.strip():
            stripped_passed = passed_key.strip()
            pool_keys = [k["key"] for k in OPENAI_KEYS]
            if stripped_passed not in pool_keys:
                if st.session_state.get("custom_openai_key", "") != stripped_passed:
                    st.session_state["custom_openai_key"] = stripped_passed
                    st.session_state["custom_openai_key_status"] = "Active"

        custom_key = st.session_state.get("custom_openai_key", "")
        custom_status = st.session_state.get("custom_openai_key_status", "Active")
        
        if custom_key and custom_status == "Active":
            return custom_key

        statuses = st.session_state["openai_keys_status"]
        idx = st.session_state["openai_current_key_index"]
        
        for i in range(len(statuses)):
            check_idx = (idx + i) % len(statuses)
            if statuses[check_idx]["status"] == "Active":
                st.session_state["openai_current_key_index"] = check_idx
                return statuses[check_idx]["key"]
                
        return OPENAI_KEYS[0]["key"]

    @staticmethod
    def mark_key_exhausted(key_val):
        """Marks the specified key as exhausted and moves to the next one."""
        is_streamlit = False
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except:
            pass

        if not is_streamlit:
            return

        OpenAIKeyManager.initialize()
        key_val_stripped = key_val.strip()
        
        custom_key = st.session_state.get("custom_openai_key", "")
        if custom_key and key_val_stripped == custom_key:
            if st.session_state["custom_openai_key_status"] != "Exhausted":
                st.session_state["custom_openai_key_status"] = "Exhausted"
                st.toast("⚠️ Custom OpenAI API Key quota exceeded! Switched to backup rotation pool.", icon="🔄")
            return

        statuses = st.session_state["openai_keys_status"]
        for idx, k in enumerate(statuses):
            if k["key"].strip() == key_val_stripped:
                if statuses[idx]["status"] != "Exhausted":
                    statuses[idx]["status"] = "Exhausted"
                    owner = k["owner"]
                    st.toast(f"⚠️ Quota exceeded for OpenAI key ({owner})! Switched to next key.", icon="🔄")
                st.session_state["openai_current_key_index"] = (idx + 1) % len(statuses)
                break

    @staticmethod
    def is_quota_error(exc):
        """Determines if the exception was caused by a quota/rate limit error, model access restriction, or network issue."""
        err_msg = str(exc).lower()
        
        # Check HTTP response body if available
        if hasattr(exc, 'response') and exc.response is not None:
            try:
                err_msg += " " + exc.response.text.lower()
            except:
                pass
                
        keywords = [
            "quota", "exhausted", "limit", "429", "rate", "insufficient_quota", 
            "billing_limit", "timeout", "timed out", "connection", "connect", "readtimedout",
            "does not exist", "invalid_value", "400", "not found", "bad request", "bad_request"
        ]
        return any(kw in err_msg for kw in keywords)

    @staticmethod
    def get_retry_delay(exc):
        """Returns the delay in seconds if transient, or None if permanent."""
        err_msg = str(exc).lower()
        if hasattr(exc, 'response') and exc.response is not None:
            try:
                err_msg += " " + exc.response.text.lower()
            except:
                pass
                
        # Permanent billing/insufficient quota/model access limits
        permanent_keywords = [
            "insufficient_quota", "billing", "quota_exceeded", "does not exist", 
            "invalid_value", "400", "not found", "bad request", "bad_request"
        ]
        if any(kw in err_msg for kw in permanent_keywords):
            return None
            
        # Check HTTP header retry-after
        if hasattr(exc, 'response') and exc.response is not None:
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except:
                    pass
                    
        # Check text body for retry patterns
        import re
        match = re.search(r"retry after ([\d\.]+)s", err_msg)
        if match:
            return float(match.group(1))
            
        if "rate limit" in err_msg or "429" in err_msg or "too many requests" in err_msg:
            return 5.0

        if "timeout" in err_msg or "timed out" in err_msg or "connection" in err_msg or "connect" in err_msg:
            return 2.0
            
        return None

def run_with_openai_rotation(api_call_fn, passed_key=None):
    """Executes an OpenAI API call function with key rotation on quota exhaustion."""
    max_attempts = len(OPENAI_KEYS) + 1
    last_error = None

    if passed_key:
        OpenAIKeyManager.get_active_key(passed_key=passed_key)

    for attempt in range(max_attempts):
        active_key = OpenAIKeyManager.get_active_key()
        
        for transient_attempt in range(3):
            try:
                return api_call_fn(active_key)
            except Exception as e:
                last_error = e
                if OpenAIKeyManager.is_quota_error(e):
                    delay = OpenAIKeyManager.get_retry_delay(e)
                    if delay is not None:
                        sleep_time = delay + 2.0
                        try:
                            st.toast(f"⏳ OpenAI Rate limit hit. Waiting {sleep_time:.1f}s to retry...", icon="⏳")
                        except:
                            print(f"[OPENAI WARNING] Rate limit hit. Waiting {sleep_time:.1f}s to retry...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        break
                else:
                    raise e
                    
        OpenAIKeyManager.mark_key_exhausted(active_key)

    raise Exception(f"All available OpenAI API keys in the rotation pool have exceeded their quota/limits. Last error: {last_error}")

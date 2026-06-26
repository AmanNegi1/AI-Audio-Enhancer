import os
import streamlit as st

# Load key rotation pool from .env or fallback to defaults
GEMINI_KEYS = []
gemini_env = os.getenv("GEMINI_KEYS_POOL")
if gemini_env:
    for item in gemini_env.split(","):
        if ":" in item:
            k, owner = item.split(":", 1)
            GEMINI_KEYS.append({"key": k.strip(), "owner": owner.strip()})


class GeminiKeyManager:
    @staticmethod
    def initialize():
        """Initializes the session state variables for key rotation."""
        # Ensure session state is initialized
        if "gemini_keys_status" not in st.session_state:
            st.session_state["gemini_keys_status"] = [
                {"key": k["key"], "owner": k["owner"], "status": "Active"} for k in GEMINI_KEYS
            ]
        if "current_key_index" not in st.session_state:
            st.session_state["current_key_index"] = 0
        if "custom_gemini_key" not in st.session_state:
            st.session_state["custom_gemini_key"] = ""
        if "custom_gemini_key_status" not in st.session_state:
            st.session_state["custom_gemini_key_status"] = "Active"

    @staticmethod
    def set_custom_key(custom_key):
        """Sets the custom user-provided API key from the UI."""
        is_streamlit = False
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except:
            pass

        if not is_streamlit:
            return

        GeminiKeyManager.initialize()
        custom_key_cleaned = custom_key.strip() if custom_key else ""
        
        # Reset custom key status to Active if the key changes
        if st.session_state.get("custom_gemini_key", "") != custom_key_cleaned:
            st.session_state["custom_gemini_key"] = custom_key_cleaned
            st.session_state["custom_gemini_key_status"] = "Active"

    @staticmethod
    def get_active_key(passed_key=None):
        """Returns the current active key (custom if available, otherwise from rotation pool)."""
        is_streamlit = False
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except:
            pass

        if not is_streamlit:
            # Standalone/CLI fallback: just return passed_key or ENV or first default key
            return passed_key if passed_key else os.environ.get("GEMINI_API_KEY", GEMINI_KEYS[0]["key"])

        GeminiKeyManager.initialize()

        # If a key is passed explicitly and is not in the rotating pool, treat it as the custom key
        if passed_key and passed_key.strip():
            stripped_passed = passed_key.strip()
            pool_keys = [k["key"] for k in GEMINI_KEYS]
            if stripped_passed not in pool_keys:
                # Update custom key if it changed
                if st.session_state.get("custom_gemini_key", "") != stripped_passed:
                    st.session_state["custom_gemini_key"] = stripped_passed
                    st.session_state["custom_gemini_key_status"] = "Active"

        # If user entered a custom key, use it if it's active
        custom_key = st.session_state.get("custom_gemini_key", "")
        custom_status = st.session_state.get("custom_gemini_key_status", "Active")
        
        if custom_key and custom_status == "Active":
            return custom_key

        # Otherwise, find the first Active key in the rotating list starting from current_key_index
        statuses = st.session_state["gemini_keys_status"]
        idx = st.session_state["current_key_index"]
        
        for i in range(len(statuses)):
            check_idx = (idx + i) % len(statuses)
            if statuses[check_idx]["status"] == "Active":
                st.session_state["current_key_index"] = check_idx
                return statuses[check_idx]["key"]
                
        # If all keys are exhausted, return the first key in the pool as a fallback
        return GEMINI_KEYS[0]["key"]

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

        GeminiKeyManager.initialize()
        key_val_stripped = key_val.strip()
        
        # Check if the exhausted key is the custom one
        custom_key = st.session_state.get("custom_gemini_key", "")
        if custom_key and key_val_stripped == custom_key:
            if st.session_state["custom_gemini_key_status"] != "Exhausted":
                st.session_state["custom_gemini_key_status"] = "Exhausted"
                st.toast("⚠️ Custom Gemini API Key quota exceeded! Switched to backup rotation pool.", icon="🔄")
            return

        # Check rotating pool
        statuses = st.session_state["gemini_keys_status"]
        for idx, k in enumerate(statuses):
            if k["key"].strip() == key_val_stripped:
                if statuses[idx]["status"] != "Exhausted":
                    statuses[idx]["status"] = "Exhausted"
                    owner = k["owner"]
                    st.toast(f"⚠️ Quota exceeded for key ({owner})! Switched to next active key.", icon="🔄")
                # Advance pointer to next index
                st.session_state["current_key_index"] = (idx + 1) % len(statuses)
                break

    @staticmethod
    def is_quota_error(exc):
        """Determines if the exception was caused by a quota/rate limit error or lack of access/paid plan."""
        err_msg = str(exc).lower()
        keywords = [
            "quota", "exhausted", "limit", "429", "rate", "resource_exhausted",
            "paid plans", "upgrade your account", "not found", "not supported"
        ]
        return any(kw in err_msg for kw in keywords)

    @staticmethod
    def get_retry_delay(exc):
        """
        Parses the exception to find if there is a retry delay.
        Returns the delay in seconds if transient, or None if it is a permanent quota limit.
        """
        err_msg = str(exc).lower()
        
        # If it's a daily limit or billing limit, it is permanent
        if "perday" in err_msg or "daily" in err_msg or "billing" in err_msg:
            return None
            
        # Check for explicit retry delay in seconds
        import re
        match = re.search(r"retry in ([\d\.]+)s", err_msg)
        if match:
            return float(match.group(1))
            
        match_seconds = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_msg)
        if match_seconds:
            return float(match_seconds.group(1))
            
        # If it mentions "retry" or is a standard 429 rate limit, default to 5 seconds
        if "retry" in err_msg or "rate limit" in err_msg or "429" in err_msg:
            return 5.0
            
        return None


def run_with_rotation(api_call_fn, passed_key=None):
    """
    Executes a Gemini API call function with automatic key rotation on quota exhaustion.
    Handles transient rate limits with backoff/sleep retries.
    """
    import time
    # Max attempts: custom key (1) + rotation pool keys
    max_attempts = len(GEMINI_KEYS) + 1
    last_error = None

    # Register passed_key as custom key first if provided
    if passed_key:
        GeminiKeyManager.get_active_key(passed_key=passed_key)

    for attempt in range(max_attempts):
        active_key = GeminiKeyManager.get_active_key()
        
        # Retry transient rate limits up to 3 times per key
        for transient_attempt in range(3):
            try:
                return api_call_fn(active_key)
            except Exception as e:
                last_error = e
                if GeminiKeyManager.is_quota_error(e):
                    # Check if it is a transient rate limit
                    delay = GeminiKeyManager.get_retry_delay(e)
                    if delay is not None:
                        sleep_time = delay + 2.0
                        try:
                            st.toast(f"⏳ Rate limit hit. Waiting {sleep_time:.1f}s to retry...", icon="⏳")
                        except:
                            print(f"[GEMINI WARNING] Rate limit hit. Waiting {sleep_time:.1f}s to retry...")
                        time.sleep(sleep_time)
                        # Loop again to retry with the same key
                        continue
                    else:
                        # Permanent limit: break the transient loop to rotate key
                        break
                else:
                    # Non-quota error, raise immediately
                    raise e
                    
        # If the transient loop finished or broke because of a permanent limit, mark key as exhausted
        GeminiKeyManager.mark_key_exhausted(active_key)

    # If all attempts fail
    raise Exception(f"All available Gemini API keys in the rotation pool have exceeded their quota/limits. Last error: {last_error}")

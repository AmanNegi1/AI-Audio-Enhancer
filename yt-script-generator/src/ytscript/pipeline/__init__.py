import os
from jinja2 import Environment, FileSystemLoader

# Resolve paths dynamically relative to this package file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))

def render_template(template_name: str, **kwargs) -> str:
    """
    Utility helper to load and render Jinja2 prompt templates.
    """
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)

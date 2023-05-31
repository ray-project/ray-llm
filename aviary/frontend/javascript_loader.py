# https://github.com/gradio-app/gradio/discussions/2932
import mimetypes
import os

import gradio.routes

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


class ScriptLoader:
    path_map = {
        "js": os.path.abspath(os.path.join(os.path.dirname(__file__), "javascript")),
        "py": os.path.abspath(os.path.join(os.path.dirname(__file__), "python")),
    }

    def __init__(self, script_type):
        self.script_type = script_type
        self.path = ScriptLoader.path_map[script_type]
        self.loaded_scripts = []

    @staticmethod
    def get_scripts(path: str, file_type: str) -> list[tuple[str, str]]:
        scripts = []
        dir_list = [os.path.join(path, f) for f in os.listdir(path)]
        files_list = [f for f in dir_list if os.path.isfile(f)]
        for s in files_list:
            # Dont forget the "." for file extension
            if os.path.splitext(s)[1] == f".{file_type}":
                scripts.append((s, os.path.basename(s)))
        return scripts


class JavaScriptLoader(ScriptLoader):
    def __init__(self):
        super().__init__("js")
        self.original_template = gradio.routes.templates.TemplateResponse
        self.load_js()
        gradio.routes.templates.TemplateResponse = self.template_response

    def load_js(self):
        js_scripts = ScriptLoader.get_scripts(self.path, self.script_type)
        for file_path, file_name in js_scripts:
            with open(file_path, "r", encoding="utf-8") as file:
                self.loaded_scripts.append(
                    f"\n<!--{file_name}-->\n<script>\n{file.read()}\n</script>"
                )

    def template_response(self, *args, **kwargs):
        response = self.original_template(*args, **kwargs)
        response.body = response.body.replace(
            "</head>".encode("utf-8"),
            f"{''.join(self.loaded_scripts)}\n</head>".encode("utf-8"),
        )
        response.init_headers()
        return response

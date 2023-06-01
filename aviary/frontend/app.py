import logging
import random
import re
import uuid

import gradio as gr
import ray
import requests
from ray import serve
from ray.serve.gradio_integrations import GradioIngress

from aviary.common.backend import get_aviary_backend
from aviary.common.constants import (
    AVIARY_DESC,
    CSS,
    EXAMPLES_IF,
    EXAMPLES_QA,
    EXAMPLES_ST,
    HEADER,
    LOGO_ANYSCALE,
    LOGO_GITHUB,
    LOGO_RAY,
    LOGO_RAY_TYPEFACE,
    MODEL_DESCRIPTION_FORMAT,
    MODEL_DESCRIPTIONS_HEADER,
    MODELS,
    NUM_LLM_OPTIONS,
    PROJECT_NAME,
    SELECTION_DICT,
    SUB_HEADER,
)
from aviary.frontend.javascript_loader import JavaScriptLoader
from aviary.frontend.leaderboard import DummyLeaderboard, Leaderboard
from aviary.frontend.mongo_secrets import get_mongo_secret_url
from aviary.frontend.utils import (
    DEFAULT_STATS,
    LOGGER,
    THEME,
    blank,
    gen_stats,
    log_flags,
    paused_logger,
    set_last,
    unset_last,
)

# Global Gradio variables
# NOTE: In the context of Gradio "global" means shared between all sessions.
BACKEND = get_aviary_backend()
ALL_MODELS = BACKEND.models()
ALL_MODELS_METADATA = {model: BACKEND.metadata(model) for model in ALL_MODELS}
MODEL_DESCRIPTIONS = (
    MODEL_DESCRIPTIONS_HEADER
    + "\n\n"
    + "\n\n".join(
        [
            MODEL_DESCRIPTION_FORMAT.format(
                model_id=k,
                model_description=v["metadata"]["model_config"]["model_description"],
                model_url=v["metadata"]["model_config"]["model_url"],
            )
            for k, v in ALL_MODELS_METADATA.items()
        ]
    )
).strip()
LLM_VALUES = [None] * NUM_LLM_OPTIONS
MONGODB_URL = get_mongo_secret_url()
LDR = Leaderboard(MONGODB_URL, PROJECT_NAME) if MONGODB_URL else DummyLeaderboard()


def gen_leaderboard():
    return LDR.generate_votes_leaderboard(), LDR.generate_perf_leaderboard()


@ray.remote(num_cpus=0)
def completions(prompt, llm, index):
    try:
        out = BACKEND.completions(prompt=prompt, llm=llm)
    except Exception as e:
        if isinstance(e, requests.ReadTimeout) or (
            hasattr(e, "response")
            and ("timeout" in e.response or e.response.status_code in (408, 504))
        ):
            out = (
                "[AVIARY] The request timed out. This usually means the server "
                "is experiencing a higher than usual load. "
                "Please try again in a few minutes."
            )
        elif hasattr(e, "response"):
            out = (
                f"[AVIARY] Backend returned an error. "
                f"Status code: {e.response.status_code}"
                f"\nResponse: {e.response.text.split('raise ')[-1]}"
            ).replace("\n", " ")
        else:
            out = f"[AVIARY] An error occurred. Please try again.\nError: {e}"
        out = {"error": out}
    return out, index


def do_query(prompt, model1, model2, model3, unused_raw=None):
    try:
        models = [model1, model2, model3]
        not_ready = [
            completions.remote(prompt, model, i) for i, model in enumerate(models)
        ]
        text_output = [""] * len(models)
        stats = [""] * len(models)
        outs = [{}] * len(models)
        while not_ready:
            ready, not_ready = ray.wait(not_ready)
            out, index = ray.get(ready[0])
            if "error" not in out:
                outs[index] = out
                text_output[index] = out["generated_text"]
                stats[index] = gen_stats(out)
            else:
                text_output[index] = out["error"]

        return [*text_output, *stats, "", outs]
    except Exception as e:
        raise gr.Error(f"An error occurred. Please try again.\nError: {e}") from e


def show_results(buttons, llm_text_boxes, llm_stats):
    for i in range(NUM_LLM_OPTIONS):
        with gr.Row(variant="compact"), gr.Column(
            variant="compact", elem_classes="output-container"
        ):
            with gr.Row():
                gr.Markdown(f"### LLM #{i + 1}")
                buttons[i] = gr.Button(
                    value=f"\U0001F31F Best answer is #{i + 1}",
                    elem_classes="rank-button pill-button",
                )
            with gr.Row(elem_classes="output-content"):
                llm_text_boxes[i] = gr.Markdown(
                    elem_classes="output-text",
                )
                llm_stats[i] = gr.Markdown(DEFAULT_STATS, elem_classes="output-stats")


def show_examples(prompt):
    with gr.Column(elem_id="prompt-examples-column"):
        gr.Examples(
            inputs=prompt,
            examples=EXAMPLES_QA,
            elem_id="prompt-examples-qa",
            label="Examples (Question Answering)",
        )
        gr.Examples(
            inputs=prompt,
            examples=EXAMPLES_IF,
            elem_id="prompt-examples-if",
            label="Examples (Instruction Following)",
        )
        gr.Examples(
            inputs=prompt,
            examples=EXAMPLES_ST,
            elem_id="prompt-examples-st",
            label="Examples (Story Telling)",
        )


def update_selection(button, choice_1, choice_2, choice_3):
    llm_choices = [choice_1, choice_2, choice_3]
    for i in range(NUM_LLM_OPTIONS):
        if button != "\U0001f3b2 Random":
            llm_choices[i] = SELECTION_DICT[button][i]
        else:
            llm_choices[i] = random.choice(ALL_MODELS)
    return llm_choices


def model_selection():
    with gr.Row(elem_id="model-dropdpown-row"):
        model_list = [x for _, x in sorted(zip(MODELS.keys(), ALL_MODELS))]
        llm_choices = [
            gr.Dropdown(
                choices=ALL_MODELS,
                value=model_list[i % len(model_list)],
                interactive=True,
                label=f"LLM #{i + 1}:",
                elem_id=f"llm-{i + 1}",
                elem_classes="llm-selector",
            )
            for i in range(NUM_LLM_OPTIONS)
        ]

    with gr.Row(elem_id="model-category-row"):
        gr.HTML("<span>Select LLMs for me: </span>")
        choices = list(SELECTION_DICT.keys())

        emoji_regex = r"\S+\s+"
        category_buttons = [
            gr.Button(
                value=choice,
                variant="secondary",
                elem_classes=f"pill-button llm-express-button llm-{re.sub(emoji_regex, '', choice).lower()}",
            )
            for choice in choices
        ]

        for cb in category_buttons:
            cb.click(
                fn=update_selection,
                inputs=[cb] + llm_choices,
                outputs=llm_choices,
            )

    return llm_choices


def create_session_id():
    return str(uuid.uuid4())


def gradio_app_builder():
    JavaScriptLoader()
    with gr.Blocks(
        theme=THEME,
        css=CSS,
        elem_id="container",
        title="Aviary Explorer",
        analytics_enabled=False,
    ) as demo:
        llm_text_boxes = [None] * NUM_LLM_OPTIONS
        llm_stats = [None] * NUM_LLM_OPTIONS
        btns = [None] * NUM_LLM_OPTIONS
        session_id = gr.State(create_session_id)
        # We need to store the raw completion
        raw_completions = gr.State([None] * NUM_LLM_OPTIONS)
        with gr.Row(elem_classes="header"):
            gr.HTML(
                f"<h1><span>{HEADER}</span><a class='logo-github' href='https://github.com/ray-project/aviary/' target='_blank'>{LOGO_GITHUB}</a></h1>",
                elem_classes="header-main",
            )
            gr.Markdown(SUB_HEADER, elem_classes="header-sub")
            gr.HTML(
                f"""<a class='logo-anyscale' href='https://www.anyscale.com' target='_blank'><span>Hosted on</span>{LOGO_ANYSCALE}</a>
                <div><span>|</span></div>
                <a class='logo-ray' href='https://ray.io' target='_blank'><span>Powered by</span>{LOGO_RAY}{LOGO_RAY_TYPEFACE}</a>
                """,
                elem_classes="branding-container",
            )
            gr.HTML(
                "<a href='https://bit.ly/run-aviary' target='_blank' id='deploy-button'>Deploy your LLMs</a>",
                elem_classes="ref-link primary",
            )
        with gr.Tab("Compare", elem_id="top-tab-group"), gr.Row():
            with gr.Column(elem_id="left-column"):
                with gr.Column(elem_id="left-column-content"):
                    llm_choices = model_selection()
                    prompt = gr.TextArea(label="Prompt", lines=5, elem_id="prompt")
                    with gr.Row():
                        clr = gr.Button(
                            value="Clear",
                            variant="secondary",
                            elem_id="clear-button",
                            elem_classes="pill-button",
                        )
                        clr.click(fn=blank, outputs=prompt)
                        submit_btn = gr.Button(
                            value="Submit",
                            variant="primary",
                            elem_id="submit-button",
                            elem_classes="pill-button",
                        )
                    show_examples(prompt)
            with gr.Column(elem_id="right-column"):
                with gr.Column(elem_id="right-column-content"):
                    show_results(btns, llm_text_boxes, llm_stats)

        with gr.Tab("Leaderboard", elem_id="leaderboard-tab"), gr.Column():
            refresh_btn = gr.Button(
                value="Refresh stats",
                variant="primary",
                elem_classes="pill-button",
                elem_id="refresh-leaderboard-button",
            )
            with gr.Column():
                gr.Markdown("## Quality")
                votes_df = gr.DataFrame(value=LDR.generate_votes_leaderboard)
            with gr.Column():
                gr.Markdown("## Performance")
                perf_df = gr.DataFrame(value=LDR.generate_perf_leaderboard)

        with gr.Tab("Models", elem_id="models-tab"), gr.Row(
            elem_id="aviary-model-desc"
        ), gr.Row():
            gr.Markdown(MODEL_DESCRIPTIONS)

        with gr.Tab("About"), gr.Row(elem_id="aviary-desc"):
            gr.Markdown(value=AVIARY_DESC)

        last_btn = gr.Textbox(visible=False)
        inputs = [prompt] + llm_choices

        if MONGODB_URL:
            LOGGER.setup(inputs + llm_text_boxes + [last_btn])
        else:
            LOGGER.setup(inputs + llm_text_boxes + [last_btn], "results")

        refresh_btn.click(fn=gen_leaderboard, outputs=[votes_df, perf_df])

        onSubmitClick = """
        function onSubmitClick(prompt, model1, model2, model3, unused_raw) {
            const element = document.querySelector('#right-column');
            if(element) {
                element.scrollIntoView({behavior: "smooth"});
            }
            return [prompt, model1, model2, model3, unused_raw];
        }
        """
        submit_btn.click(
            fn=do_query,
            inputs=inputs + [raw_completions],
            outputs=llm_text_boxes + llm_stats + [last_btn, raw_completions],
            _js=onSubmitClick,
        ).then(
            fn=log_flags,
            inputs=inputs + llm_text_boxes + [last_btn, raw_completions, session_id],
        )

        for i in range(NUM_LLM_OPTIONS):
            btns[i].click(
                fn=set_last, inputs=btns[i], outputs=[last_btn, btns[i]]
            ).then(
                lambda *args: paused_logger(args),
                inputs=inputs
                + llm_text_boxes
                + [last_btn, raw_completions, session_id],
            ).then(
                fn=unset_last, inputs=btns[i], outputs=btns[i]
            )
        with gr.Row(elem_id="footer"):
            gr.HTML(
                f"<a href='https://github.com/ray-project/aviary/'>{LOGO_GITHUB}<span>GitHub</span></a> • <a href='https://anyscale.com/aviary-tos' target='_blank'>Terms of Use</a> • <a href='https://anyscale.com/privacy-policy' target='_blank'>Privacy Policy</a>"
            )
        return demo


std_logger = logging.getLogger("ray.serve")


@serve.deployment
class AviaryFrontend(GradioIngress):
    def __init__(self, builder):
        # Aviary deployment simply silences the unnecessary gradio info calls.
        std_logger.setLevel(logging.ERROR)
        super().__init__(builder)


app = AviaryFrontend.options(ray_actor_options={"num_cpus": 4}, name=PROJECT_NAME).bind(
    gradio_app_builder
)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    gradio_app_builder().launch(show_error=True)

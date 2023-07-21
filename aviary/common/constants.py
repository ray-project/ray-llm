PROJECT_NAME = "AviaryFrontend"


# we add the trailing slash to make it easier to construct URLs and test
# for backwards compatibility
DEFAULT_API_VERSION = ""

NUM_LLM_OPTIONS = 3

# AWS timeout
# TODO make this configurable
GATEWAY_TIMEOUT_S = 110

# (connect, read) timeouts in seconds. we make the "read" timeout deliberately
# shorter than in cloudfront OR gradio, so that we can explicitly handle timeouts.
TIMEOUT = (5, GATEWAY_TIMEOUT_S - 5)

AVIARY_DESC = """

# What is this?

Aviary Explorer allows you to take a single prompt and send it to a number of open source LLMs
hosted by Anyscale. It also gives you estimates of the cost as well as the latency.

It is built on top of [Ray](https://ray.io) by the company
[Anyscale](https://anyscale.com).

It is an open source project and you can launch your own instance of aviary.backend.

If you would like to use a managed version of Aviary Explorer specific to your company,
please let aviary@anyscale.com know.

# Notes

## LLM pre-selection buttons

In the interface there are some buttons to preselect three different LLMs. 

- *Fast*: This chooses the LLMs with the lowest latency. 
- *Strong*: This chooses the LLMs that are consistently at the top of the leaderboard. 
- *Variant*: This shows three different variants of the same model (MosaicML's mpt-7b) with different fine tuning applied. 
- *Random*: Selects 3 LLMs _with replacement_ for you to compare. 

## Win Ratio

The "win Ratio" shows the relative performance of different LLMs. Higher is  better, with the perfect score (beats all other algorithms 100 per cent of the time) being 3000. If you were to randomly pick which answer is best, that would result in awin ratio of 1000.

## Cost

The cost is calculated assuming it is using an on-demand AWS g5.xlarge instance at
list prices.
It also doesn't include the benefits of batching (which can't be guaranteed).
Hence it can be considered an upper bound on the cost.
"""

EXAMPLES_QA = [
    "How do I make fried rice? ",
    "What are the 5 best sci fi books? ",
    "What are the best places in the world to visit? ",
    "Which Olympics were held in Australia?",
]

EXAMPLES_IF = [
    "Please describe a beautiful house.",
    "Generate 5 second grade level math problems.",
    "Write a poem about shoes.",
]

EXAMPLES_ST = [
    "Once upon a time, ",
    "It was the worst of times, it was the best of times  ",
    "To be or not to be, ",
]

MODELS = {
    "CarperAI/stable-vicuna-13b-delta": "This is a model based on Vicuna but with reinforcement learning with human "
    "feedback applied. Pretty Good. Based on LLaMa so non-commercial use only. "
    "13B parameters.",
    "lmsys/vicuna-13b-delta-v1.1": "LLaMA with additional training on records from ChatGPT. One of the best. "
    "Based on LLaMa so non-commercial use only. 13B Parameters.",
    "stabilityai/stablelm-tuned-alpha-7b": "Model trained with large window (4096 tokends). Works OK for summarization. "
    "Tuned for instruction following. CC-BY-SA license.",
    "mosaicml/mpt-7b-instruct": "TODO",
    "amazon/LightGPT": "TODO",
    "databricks/dolly-v2-12b": "TODO",
    "RWKV/rwkv-raven-14b": "TODO",
    "mosaicml/mpt-7b-chat": "TODO",
    "mosaicml/mpt-7b-storywriter": "TODO",
    "h2oai/h2ogpt-oasst1-512-12b": "TODO",
    "OpenAssistant/oasst-sft-7-llama-30b-xor": "TODO",
}

SELECTION_DICT = {
    "\U0001f680 Fast": [
        "amazon/LightGPT",
        "stabilityai/stablelm-tuned-alpha-7b",
        "mosaicml/mpt-7b-chat",
    ],
    "\U0001f4aa Strong": [
        "CarperAI/stable-vicuna-13b-delta",
        "OpenAssistant/oasst-sft-7-llama-30b-xor",
        "mosaicml/mpt-7b-chat",
    ],
    "\U0001f46a Variants": [
        "mosaicml/mpt-7b-instruct",
        "mosaicml/mpt-7b-chat",
        "mosaicml/mpt-7b-storywriter",
    ],
    "\U0001f3b2 Random": [],
}

MODEL_DESCRIPTIONS_HEADER = "# Model Descriptions\n\n"

MODEL_DESCRIPTION_FORMAT = "## [{model_id}]({model_url})\n{model_description}"

HEADER = """\U0001f99c \U0001f50d Aviary Explorer"""

SUB_HEADER = """A place to study stochastic parrots"""

CSS = """
::-webkit-scrollbar {
  width: 4px;
}
::-webkit-scrollbar-track {
  border-radius: 2px;
}
::-webkit-scrollbar-thumb {
  background: var(--button-primary-background-fill);
  border-radius: 4px;
}
body, gradio-app {
    height: 100vh;
    max-height: 100vh;
}
.pill-button {
    border-radius: 32px !important;
    border: none !important;
}

.block.ref-link {
    flex-grow: 0 !important;
}
.block.ref-link a {
    background: none !important;
    border: 2px solid var(--button-secondary-border-color) !important;
    border-radius: 20px;
    color: var(--button-primary-text-color) !important;
    display: block;
    font-weight: bold;
    height: 40px;
    line-height: 36px;
    text-decoration: none;
    text-align: center;
}
.block.ref-link a:visited {
    color: var(--button-primary-text-color) !important;
}
.block.ref-link.primary a {
    background: var(--button-primary-background-fill) !important;
    border: 2px solid var(--button-primary-background-fill) !important;
}
.main,
.contain,
#component-0,
#top-tab-group > div,
#top-tab-group > div > div,
#left-column,
#right-column,
.output-text > label,
.output-text > label > textarea {
    height: 100% !important;
}

.contain,
#component-0,
#component-0 > .tabs,
#top-tab-group,
#top-tab-group > div {
    overflow: hidden !important;
}

@media only screen and (max-width: 768px) {
    .gradio-container,
    .contain,
    #component-0,
    #component-0 > .tabs,
    #top-tab-group,
    #top-tab-group > div {
        overflow: auto !important;
    }
    .main,
    .contain,
    #component-0,
    #top-tab-group > div,
    #top-tab-group > div > div,
    #left-column,
    #right-column {
        height: fit-content !important;
    }
}
.contain > .gap {
    gap: 8px !important;
}

.contain > .gap > .header {
    align-items: center;
    flex-direction: row;
}
.contain > .gap > .header > .header-main {
    flex-basis: fit-content;
    flex-grow: 0;
}
.header-main > h1 {
    align-items: center;
    display: flex;
    flex: 0 0 fit-content;
    flex-direction: row;
    gap: 8px;
}
@media only screen and (max-width: 1024px) {
    .contain > .gap > .header > .header-main {
        flex-grow: 1;
    }
}
.header-main > h1 a {
    text-decoration: none !important;
}
.header-main .logo-github svg {
}
.header-main .logo-github path {
    fill: var(--button-primary-text-color) !important;
}
.contain > .gap > .header > .header-sub {
    flex-grow: 1;
}
@media only screen and (max-width: 768px) {
    .contain > .gap > .header {
        align-items: center;
        flex-direction: column;
        margin-bottom: 32px;
    }
}
@media only screen and (max-width: 1024px) {
    .branding-container,
    .contain > .gap > .header > .header-sub {
        display: none !important;
    }
}
@media only screen and (max-width: 768px) {
    .branding-container {
        display: flex !important;
    }
}

.branding-container {
    align-items: center;
    display: flex;
    flex: 0 0 fit-content !important;
    flex-direction: row;
    gap: 8px;
    height: 100%;
}
.branding-container a,
.branding-container a:visited {
    color: var(--body-text-color-subdued) !important;
    text-decoration: none !important;
}
.branding-container span {
    color: var(--body-text-color-subdued) !important;
    line-height: 22px !important;
}

.logo-anyscale,
.logo-ray {
    white-space: nowrap;
}
.logo-anyscale > svg,
.logo-ray > svg {
    display: inline-block;
    height: 20px;
    margin-left: 8px;    
}
.logo-anyscale > svg {
    width: 88px;
}
.logo-ray > .ray-icon {
    width: 20px;
}
.logo-ray > .ray-typeface {
    height: 10px;
    width: 28px;
}
@media (prefers-color-scheme: dark) {
    .logo-anyscale > svg > path,
    .logo-ray > svg > path {
        fill: #fff !important;
    }
}
@media (prefers-color-scheme: light) {
    .logo-anyscale > svg > path {
        fill: #234999 !important;
    }
    .logo-ray > .ray-icon > path {
        fill: #00AEEF !important;
    }
    .logo-ray > .ray-typeface > path {
        fill: #000 !important;
    }
}
.contain > .gap > .tabs {
    display: flex;
    flex-direction: column;
}
#top-tab-group {
    padding: 16px !important;
}
#top-tab-group > div > div {
    gap: 48px !important;
}
.tabs,
.tabitem {
    flex-grow: 1 !important;
}

#left-column,
#right-column {
    overflow: auto;
    padding-right: 12px;
}
@media only screen and (max-width: 768px) {
    #left-column,
    #right-column {
        padding-right: 0px;
    }
}

#left-column-content,
#right-column-content {
    flex-wrap: nowrap;
}

#prompt,
#left-column-content > .form,
.llm-selector {
    background: none !important;
    border: none !important;
    padding: 0 !important;
}
#left-column-content label > span {
    font-weight: bold;
    font-size: 1rem;
}

.ticker-container.block {
    padding: 4px 8px !important;
    border: 1px solid var(--button-primary-border-color) !important;
}

#prompt-examples-column {
    flex-grow: 0 !important;
}
#prompt-examples-column .gallery > button {
    opacity: .5;
}
#prompt-examples-column .gallery > button:hover {
    opacity: 1;
}

#right-column-content {
    gap: 24px;
}
#right-column-content > .compact,
#right-column-content > .compact > .compact,
#right-column-content > .compact > .compact > .block {
    background: none !important;
    border: none !important;
    padding: 0 !important;
}
#right-column-content > div {
    flex-grow: 1 !important;
}

.output-container {
    gap: 8px !important;
    position: relative;
}
.llm-express-button,
.rank-button {
    flex-grow: 0 !important;
    font-size: .8rem !important;
    font-weight: normal !important;
    line-height: 16px !important;
    min-width: fit-content !important;
    padding: 4px 12px !important;
    white-space: nowrap;
}
.output-content {
    flex-grow: 1 !important;
    flex-direction: row;
}
@media only screen and (max-width: 768px) {
    .output-content {
        flex-direction: column;
    }
}
.output-text.block {
    background: var(--input-background-fill) !important;
    border-width: 1px !important;
    flex-grow: 3 !important;
    padding: 8px 12px !important;
}
.output-content > .output-stats {
    flex-grow: 1 !important;
}
.output-stats > table {
    width: 100% !important;
}
.output-stats td,
.output-stats th {
    color: var(--body-text-color-subdued) !important;
    padding: 0 !important;
}
.output-stats th,
.output-stats td {
    border-bottom-color: var(--body-text-color-subdued) !important;
}

#leaderboard-tab,
#models-tab {
    overflow: auto !important;
}
#refresh-leaderboard-button {
    margin-top: 12px;
}
footer {
    display: none !important;
}
#footer {
    color: var(--body-text-color-subdued) !important;
    text-align: center;
}
#footer a,
#footer a:visited,
#footer span {
    color: var(--body-text-color-subdued) !important;
    text-decoration: none !important;
}
#footer svg {
    display: inline-block;
    height: 18px;
    vertical-align: text-top;
    width: 18px;
}
#footer svg path {
    fill: var(--body-text-color-subdued) !important;
}
"""

LOGO_ANYSCALE = """
<svg width="595" height="136" viewBox="0 0 595 136" fill="none" xmlns="http://www.w3.org/2000/svg"> <rect width="594.84" height="135.359" fill="none"/> <path d="M45.0805 73.9193H16.4409C12.0946 73.9299 7.92931 75.6611 4.85603 78.7344C1.78275 81.8076 0.0515777 85.9728 0.0410156
90.3191V118.959C0.0515777 123.305 1.78275 127.47 4.85603 130.544C7.92931 133.617 12.0946 135.348 16.4409 135.359H45.0805C49.4267 135.348 53.592 133.617 56.6653 130.544C59.7386 127.47 61.4698 123.305 61.4803 118.959V90.3191C61.4698 85.9728 59.7386 81.8076 56.6653 78.7344C53.592 75.6611 49.4267 
73.9299 45.0805 73.9193ZM51.4405 118.959C51.4405 119.794 51.276 120.621 50.9564 121.393C50.6368 122.164 50.1683 122.865 49.5777 123.456C48.9872 124.047 48.286 124.515 47.5144 124.835C46.7428 125.154 45.9157 125.319 45.0805 125.319H16.4409C15.6057 125.319 14.7786 125.154 14.0069 124.835C13.2353
124.515 12.5343 124.047 11.9437 123.456C11.3532 122.865 10.8846 122.164 10.565 121.393C10.2453 120.621 10.0808 119.794 10.0808 118.959V90.3191C10.0808 88.6323 10.751 87.0147 11.9437 85.8219C13.1364 84.6292 14.7541 83.9592 16.4409 83.9592H45.0805C46.7672 83.9592 48.385 84.6292 49.5777
85.8219C50.7705 87.0147 51.4405 88.6323 51.4405 90.3191V118.959Z" fill="#234999"/> <path d="M119.039 50.8795H84.4791V16.3999C84.4686 12.0536 82.7374 7.88835 79.6641 4.81507C76.5908 1.7418 72.4255 0.010562 68.0793 0H16.3999C12.0536 0.010562 7.88841 1.7418 4.81514 4.81507C1.74186 7.88835
0.010562 12.0536 0 16.3999V44.3996C0.010562 48.7458 1.74186 52.9111 4.81514 55.9844C7.88841 59.0576 12.0536 60.7889 16.3999 60.7994H74.4393V118.799C74.4499 123.145 76.181 127.31 79.2543 130.384C82.3276 133.457 86.4928 135.188 90.839 135.199H118.839C123.185 135.188 127.35 133.457 130.424
130.384C133.497 127.31 135.228 123.145 135.239 118.799V67.2793C135.229 62.9674 133.525 58.832 130.495 55.7644C127.464 52.6967 123.35 50.9424 119.039 50.8795ZM16.3999 50.8795C15.5545 50.8796 14.7175 50.7113 13.938 50.3842C13.1584 50.0571 12.4519 49.5778 11.8597 48.9745C11.2676 48.3711 10.8016
47.6558 10.4892 46.8703C10.1767 46.0848 10.024 45.2448 10.0399 44.3996V16.3999C10.0399 14.7131 10.71 13.0954 11.9027 11.9027C13.0954 10.71 14.7131 10.0399 16.3999 10.0399H68.0393C68.8779 10.0347 69.7092 10.1953 70.4855 10.5125C71.2617 10.8298 71.9676 11.2974 72.5624 11.8885C73.1572
12.4796 73.6293 13.1825 73.9514 13.9568C74.2735 14.731 74.4393 15.5613 74.4393 16.3999V50.9195H16.3999V50.8795ZM125.399 118.879C125.399 120.566 124.729 122.183 123.536 123.376C122.343 124.569 120.726 125.239 119.039 125.239H91.0391C89.3523 125.239 87.7347 124.569 86.542 123.376C85.3492
122.183 84.6792 120.566 84.6792 118.879V60.8794H119.239C120.926 60.8794 122.543 61.5494 123.736 62.7422C124.929 63.9349 125.599 65.5526 125.599 67.2394L125.399 118.879Z" fill="#234999"/> <path d="M335.957 74.0793L323.237 39.5996H309.237L329.037 87.7191L318.317 112.319H331.597L362.157
39.5996H348.877L335.957 74.0793Z" fill="#234999"/> <path d="M396.037 62.5994C393.158 61.4737 390.204 60.5514 387.197 59.8394C384.524 59.2368 381.923 58.3518 379.437 57.1994C378.687 56.8918 378.049 56.3637 377.606 55.6852C377.163 55.0066 376.936 54.2095 376.957 53.3995C376.937 52.71 377.097 52.0273
377.419 51.4175C377.741 50.8078 378.216 50.2917 378.797 49.9195C380.517 49.0225 382.443 48.5946 384.381 48.6788C386.319 48.7631 388.201 49.3566 389.837 50.3995C390.549 50.9647 391.134 51.6735 391.555 52.4798C391.975 53.2861 392.221 54.1719 392.277 55.0795H404.276C404.206 52.7842
403.653 50.5296 402.654 48.4621C401.655 46.3946 400.231 44.5607 398.476 43.0796C394.461 39.9747 389.465 38.4135 384.397 38.6797C380.831 38.5943 377.287 39.2622 373.997 40.6396C371.295 41.7736 368.957 43.6276 367.237 45.9995C365.687 48.1845 364.861 50.8007 364.877 53.4795C364.701
56.3657 365.599 59.215 367.397 61.4794C369.042 63.402 371.128 64.8979 373.477 65.8394C376.42 66.9733 379.442 67.8957 382.517 68.5993C385.19 69.2452 387.8 70.1287 390.317 71.2393C391.027 71.5401 391.636 72.04 392.069 72.6787C392.501 73.3175 392.74 74.0678 392.757 74.8393C392.756 75.5652
392.572 76.2792 392.223 76.9154C391.873 77.5515 391.369 78.0894 390.757 78.4792C389.097 79.5226 387.154 80.0257 385.197 79.9192C383.003 79.9985 380.847 79.3362 379.077 78.0392C378.309 77.4885 377.664 76.7847 377.183 75.9724C376.701 75.16 376.392 74.2567 376.277 73.3193H363.717C363.889
76.3258 364.962 79.211 366.797 81.5992C368.797 84.1917 371.427 86.2296 374.437 87.5191C377.863 89.0271 381.574 89.7775 385.317 89.7191C388.839 89.791 392.338 89.1375 395.596 87.7992C398.305 86.6963 400.648 84.8525 402.356 82.4792C403.929 80.2302 404.755 77.5435 404.716 74.7993C404.832
71.9012 403.894 69.0593 402.076 66.7994C400.427 64.9379 398.356 63.4979 396.037 62.5994Z" fill="#234999"/> <path d="M435.317 49.3995C437.553 49.3296 439.752 49.9742 441.597 51.2395C443.308 52.4882 444.571 54.2558 445.197 56.2794H458.637C457.555 51.1842 454.725 46.6278 450.637
43.3996C446.228 40.2086 440.875 38.5889 435.437 38.7996C431.019 38.7192 426.658 39.8094 422.797 41.9596C419.13 44.054 416.133 47.1474 414.157 50.8795C412.103 55.0548 411.035 59.6461 411.035 64.2994C411.035 68.9527 412.103 73.5439 414.157 77.7192C416.133 81.4513 419.13 84.5447 422.797
86.6391C426.657 88.7911 431.018 89.8815 435.437 89.7991C440.879 89.9913 446.225 88.3269 450.597 85.0791C454.632 81.8548 457.452 77.3559 458.597 72.3193H445.237C444.619 74.4105 443.299 76.2246 441.499 77.456C439.699 78.6874 437.53 79.2606 435.357 79.0792C433.727 79.1326 432.109 78.7978 430.634
78.1023C429.16 77.4068 427.872 76.3706 426.877 75.0792C424.841 71.7862 423.762 67.991 423.762 64.1193C423.762 60.2476 424.841 56.4525 426.877 53.1595C427.893 51.9176 429.185 50.9308 430.651 50.2779C432.116 49.625 433.714 49.3242 435.317 49.3995Z" fill="#234999"/> <path d="M502.516
42.7996C502.517 43.0782 502.438 43.3511 502.287 43.5854C502.136 43.8197 501.921 44.0051 501.667 44.1194C501.413 44.2338 501.131 44.2719 500.856 44.2294C500.581 44.1868 500.324 44.0653 500.116 43.8796C498.901 42.7536 497.559 41.7736 496.116 40.9596C493.129 39.4106 489.8 38.6402 486.436
38.7197C482.406 38.6721 478.444 39.7528 474.996 41.8396C471.552 44.0106 468.781 47.0998 466.996 50.7595C464.939 54.879 463.91 59.4355 463.996 64.0394C463.909 68.6818 464.938 73.2772 466.996 77.4392C468.785 81.1278 471.553 84.2547 474.996 86.4792C478.398 88.6119 482.341 89.7226 486.356
89.6792C489.749 89.746 493.104 88.9622 496.116 87.3992C498.759 86.0386 501.036 84.0632 502.756 81.6392V88.8391H515.316V39.5997H502.756L502.516 42.7996ZM500.716 72.0793C499.617 74.1632 497.953 75.8958 495.916 77.0793C493.939 78.2372 491.688 78.8449 489.396 78.8392C487.151 78.8359 484.95 78.2131
483.036 77.0393C481.001 75.8002 479.342 74.0302 478.236 71.9193C476.971 69.5175 476.338 66.8333 476.396 64.1194C476.328 61.4299 476.962 58.7692 478.236 56.3995C479.061 54.9021 480.176 53.5844 481.516 52.5237C482.856 51.463 484.395 50.6806 486.042 50.2224C487.688 49.7641 489.41 49.6392
491.106 49.8551C492.801 50.071 494.437 50.6233 495.916 51.4795C497.952 52.6647 499.615 54.3969 500.716 56.4795C501.889 58.9124 502.498 61.5785 502.498 64.2794C502.498 66.9803 501.889 69.6464 500.716 72.0793Z" fill="#234999"/> <path d="M237.038 42.7996C237.04 43.0782 236.96 43.3511
236.809 43.5854C236.659 43.8197 236.443 44.0051 236.189 44.1194C235.935 44.2338 235.654 44.2719 235.378 44.2294C235.103 44.1868 234.846 44.0653 234.638 43.8796C233.423 42.7539 232.081 41.774 230.638 40.9596C227.652 39.4106 224.322 38.6402 220.958 38.7197C216.929 38.6721 212.966 39.7528
209.519 41.8396C206.074 44.0106 203.303 47.0998 201.519 50.7595C199.461 54.879 198.432 59.4355 198.519 64.0394C198.432 68.6818 199.46 73.2772 201.519 77.4392C203.31 81.1267 206.076 84.2532 209.519 86.4792C212.921 88.6119 216.864 89.7226 220.879 89.6792C224.271 89.746 227.627 88.9622
230.638 87.3992C233.281 86.0386 235.558 84.0632 237.278 81.6392V88.8391H249.638V39.5997H237.078L237.038 42.7996ZM235.238 72.0793C234.139 74.1632 232.476 75.8958 230.438 77.0793C228.461 78.2372 226.21 78.8449 223.919 78.8392C221.673 78.8359 219.473 78.2131 217.559 77.0393C215.523
75.8002 213.864 74.0302 212.759 71.9193C211.494 69.5175 210.86 66.8333 210.919 64.1194C210.851 61.4299 211.485 58.7692 212.759 56.3995C213.583 54.9021 214.698 53.5844 216.038 52.5237C217.379 51.463 218.917 50.6806 220.564 50.2224C222.211 49.7641 223.933 49.6392 225.628 49.8551C227.324
50.071 228.959 50.6233 230.438 51.4795C232.474 52.6647 234.137 54.3969 235.238 56.4795C236.411 58.9124 237.021 61.5785 237.021 64.2794C237.021 66.9803 236.411 69.6464 235.238 72.0793Z" fill="#234999"/> <path d="M537.836 22.9998H525.356V88.9591H537.836V22.9998Z" fill="#234999"/> <path
d="M594.835 63.1994C594.923 58.7392 593.862 54.3312 591.755 50.3995C589.773 46.7531 586.77 43.7642 583.115 41.7996C579.165 39.7776 574.792 38.7232 570.355 38.7232C565.918 38.7232 561.544 39.7776 557.595 41.7996C553.88 43.8582 550.849 46.9589 548.875 50.7195C546.649 54.9297 545.534 59.6381
545.635 64.3994C545.539 69.0485 546.627 73.6456 548.795 77.7592C550.819 81.5334 553.889 84.6442 557.635 86.7191C561.555 88.865 565.966 89.9539 570.435 89.8791C575.768 90.0425 581.004 88.424 585.315 85.2792C589.251 82.3695 592.13 78.2545 593.515 73.5592H580.035C579.182 75.434 577.782
77.0067 576.019 78.0705C574.255 79.1343 572.211 79.6394 570.155 79.5192C567.197 79.5707 564.332 78.4822 562.155 76.4793C559.866 74.2848 558.518 71.2878 558.395 68.1193H594.395C594.678 66.4944 594.825 64.8487 594.835 63.1994ZM558.475 59.6394C558.763 56.5669 560.19 53.7133 562.475
51.6395C564.678 49.7054 567.546 48.7016 570.475 48.8395C573.495 48.7519 576.441 49.7768 578.755 51.7195C579.869 52.6697 580.758 53.8548 581.36 55.1895C581.962 56.5242 582.26 57.9755 582.235 59.4394L558.475 59.6394Z" fill="#234999"/> <path d="M287.197 38.8796C284.256 38.8511 281.345 39.4798
278.677 40.7196C277.091 41.4604 275.621 42.4308 274.317 43.5996C274.143 43.7556 273.927 43.8572 273.696 43.892C273.464 43.9269 273.228 43.8933 273.016 43.7956C272.803 43.6979 272.624 43.5402 272.5 43.3419C272.376 43.1437 272.312 42.9135 272.317
42.6796V39.5997H259.837V88.9991H272.317V61.7194C272.105 58.4308 273.181 55.1885 275.317 52.6795C276.359 51.627 277.608 50.8025 278.985 50.2585C280.362 49.7146 281.838 49.4629 283.317 49.5195C284.799 49.4452 286.279 49.6887 287.659 50.2337C289.039 50.7787 290.286 51.6127 291.317 52.6795C293.428 
55.2007 294.488 58.4378 294.277 61.7194V88.9991H306.757V59.9994C306.977 57.1898 306.638 54.3645 305.759 51.6869C304.88 49.0092 303.479 46.5324 301.637 44.3996C299.75 42.5209 297.49 41.059 295.003 40.1082C292.515 39.1574 289.856 38.7388 287.197 38.8796Z" fill="#234999"/> </svg>

"""
LOGO_RAY = """
<svg class="ray-icon" width="224" height="224" viewBox="0 0 224 224" fill="none" xmlns="http://www.w3.org/2000/svg"> <path d="M82.3502 104.44C84.0188 97.8822 87.825 92.0673 93.1673 87.9141C98.5096 83.7609 105.083 81.5062 111.85 81.5062C118.617 81.5062 125.191 83.7609 130.533
87.9141C135.875 92.0673 139.682 97.8822 141.35 104.44H163.75C164.476 101.561 165.622 98.8053 167.15 96.26L127.45 56.56C121.071 60.3517 113.527 61.6818 106.236 60.3C98.9445 58.9183 92.4096 54.9198 87.8604 49.057C83.3111 43.1941 81.061 35.871 81.5334 28.4652C82.0058 21.0594 85.1681 14.0815
90.4253 8.84414C95.6826 3.6068 102.673 0.47105 110.08 0.0268077C117.488 -0.417435 124.802 1.86045 130.648 6.43194C136.493 11.0034 140.467 17.5534 141.821 24.8496C143.175 32.1459 141.816 39.6855 138 46.05L177.69 85.75C182.31 82.9873 187.58 81.499 192.963 81.437C198.345 81.375 203.648
82.7415 208.33 85.3971C213.013 88.0527 216.907 91.9025 219.616 96.5539C222.326 101.205 223.753 106.492 223.753 111.875C223.753 117.258 222.326 122.545 219.616 127.196C216.907 131.848 213.013 135.697 208.33 138.353C203.648 141.008 198.345 142.375 192.963 142.313C187.58 142.251 182.31
140.763 177.69 138L138 177.7C141.808 184.071 143.156 191.614 141.79 198.909C140.424 206.205 136.44 212.75 130.586 217.312C124.732 221.875 117.412 224.14 110.004 223.683C102.596 223.225 95.6105 220.076 90.3623 214.828C85.114 209.58 81.9649 202.594 81.5074 195.186C81.0499 187.778 83.3156
180.459 87.8781 174.605C92.4407 168.75 98.9855 164.766 106.281 163.4C113.576 162.035 121.119 163.382 127.49 167.19L167.19 127.49C165.664 124.94 164.518 122.181 163.79 119.3H141.39C139.722 125.858 135.915 131.673 130.573 135.826C125.231 139.979 118.657 142.234 111.89 142.234C105.123
142.234 98.5496 139.979 93.2073 135.826C87.865 131.673 84.0588 125.858 82.3902 119.3H60.0002C58.188 126.495 53.8087 132.779 47.6865 136.971C41.5643 141.162 34.1213 142.972 26.7581 142.059C19.3949 141.146 12.6193 137.573 7.70623 132.013C2.7932 126.453 0.081543 119.29 0.081543
111.87C0.081543 104.45 2.7932 97.2866 7.70623 91.7268C12.6193 86.1669 19.3949 82.5943 26.7581 81.6812C34.1213 80.7682 41.5643 82.5776 47.6865 86.7691C53.8087 90.9606 58.188 97.2451 60.0002 104.44H82.3502ZM100.86 204.32C103.408 206.868 106.76 208.453 110.345 208.806C113.93 209.159 117.527
208.257 120.522 206.256C123.517 204.254 125.726 201.275 126.771 197.827C127.816 194.38 127.633 190.676 126.253 187.348C124.874 184.02 122.383 181.274 119.205 179.577C116.027 177.879 112.359 177.337 108.826 178.041C105.293 178.746 102.113 180.653 99.8293 183.439C97.5453 186.225 96.2981
189.717 96.3002 193.32C96.2987 195.363 96.7008 197.387 97.4833 199.275C98.2658 201.162 99.4134 202.877 100.86 204.32ZM204.32 122.88C206.868 120.333 208.453 116.981 208.806 113.395C209.159 109.81 208.258 106.213 206.256 103.218C204.254 100.223 201.275 98.0146 197.828 96.9695C194.38
95.9244 190.676 96.1073 187.348 97.4869C184.02 98.8665 181.274 101.357 179.577 104.535C177.88 107.713 177.337 111.381 178.041 114.914C178.746 118.447 180.654 121.627 183.44 123.911C186.226 126.195 189.718 127.442 193.32 127.44C195.364 127.443 197.388 127.041 199.276 126.258C201.163
125.476 202.878 124.328 204.32 122.88ZM122.88 19.42C120.333 16.8725 116.981 15.2872 113.396 14.9342C109.81 14.5813 106.214 15.4825 103.218 17.4844C100.223 19.4863 98.0148 22.4649 96.9697 25.9126C95.9247 29.3603 96.1075 33.0638 97.4871 36.3918C98.8667 39.7198 101.358 42.4664 104.536
44.1635C107.713 45.8606 111.381 46.4032 114.914 45.6988C118.448 44.9944 121.627 43.0866 123.911 40.3006C126.195 37.5145 127.442 34.0226 127.44 30.42C127.441 28.3767 127.038 26.3534 126.256 24.4659C125.473 22.5784 124.326 20.8637 122.88 19.42ZM19.4202 100.86C16.8727 103.407 15.2874 106.759
14.9344 110.345C14.5815 113.93 15.4827 117.527 17.4846 120.522C19.4865 123.517 22.4651 125.725 25.9128 126.77C29.3605 127.816 33.064 127.633 36.392 126.253C39.72 124.874 42.4666 122.383 44.1637 119.205C45.8608 116.027 46.4034 112.359 45.699 108.826C44.9946 105.293 43.0868 102.113
40.3008 99.8291C37.5147 97.5451 34.0228 96.2979 30.4202 96.3C26.294 96.3014 22.3372 97.9416 19.4202 100.86ZM100.86 100.86C98.3127 103.407 96.7274 106.759 96.3744 110.345C96.0215 113.93 96.9227 117.527 98.9246 120.522C100.927 123.517 103.905 125.725 107.353 126.77C110.801 127.816
114.504 127.633 117.832 126.253C121.16 124.874 123.907 122.383 125.604 119.205C127.301 116.027 127.843 112.359 127.139 108.826C126.435 105.293 124.527 102.113 121.741 99.8291C118.955 97.5451 115.463 96.2979 111.86 96.3C109.817 96.2985 107.793 96.7005 105.905 97.4831C104.018 98.2656
102.303 99.4132 100.86 100.86Z" fill="#00AEEF"/> </svg>
"""
LOGO_RAY_TYPEFACE = """
<svg class="ray-typeface" width="296" height="108" viewBox="0 0 296 108" fill="none" xmlns="http://www.w3.org/2000/svg"> <path d="M70.2903 107.56H84.2903L59.1302 71.05C74.6302 65.99 83.2102 54.05 83.2102 36.23C83.2102 13.37 67.7202 0.179993 40.8802
0.179993H0.000244141V107.56H12.2702V73.66H40.8002C43.2502 73.66 45.7103 73.5 48.0103 73.35L70.2502 107.56H70.2903ZM12.2903 61.85V12H40.8203C60.3003 12 71.3402 20.29 71.3402 36.55C71.3402 53.27 60.3403 61.86 40.8203 61.86L12.2903 61.85ZM180.4 80.41L192.4 107.56H205.74L157.74 0.179993H145L96.8302 
107.56H109.83L121.83 80.41H180.4ZM175.19 68.6H127L151 14.14L175.24 68.6H175.19ZM255.11 70.74L295.91 0.179993H283.35L248.99 56.18L214.17 0.179993H201.44L243.01 71.18V107.54H255.13V70.74H255.11Z" fill="#231F20"/> </svg>
"""
LOGO_GITHUB = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_8107_84105)"><path d="M12 2C6.475 2 2 6.475 2 12C1.99887 14.0993 2.65882 16.1456 3.88622 17.8487C5.11362 19.5517 6.84615 20.8251 8.838 21.488C9.338 21.575 9.525 21.275 9.525 
21.012C9.525 20.775 9.512 19.988 9.512 19.15C7 19.613 6.35 18.538 6.15 17.975C6.037 17.687 5.55 16.8 5.125 16.562C4.775 16.375 4.275 15.912 5.112 15.9C5.9 15.887 6.462 16.625 6.65 16.925C7.55 18.437 8.988 18.012 9.562 17.75C9.65 17.1 9.912 16.663 10.2 16.413C7.975 16.163 5.65 15.3 5.65 11.475C5.65 
10.387 6.037 9.488 6.675 8.787C6.575 8.537 6.225 7.512 6.775 6.137C6.775 6.137 7.612 5.875 9.525 7.163C10.3391 6.93706 11.1802 6.82334 12.025 6.825C12.875 6.825 13.725 6.937 14.525 7.162C16.437 5.862 17.275 6.138 17.275 6.138C17.825 7.513 17.475 8.538 17.375 8.788C18.012 9.488 18.4 10.375 18.4 
11.475C18.4 15.313 16.063 16.163 13.838 16.413C14.2 16.725 14.513 17.325 14.513 18.263C14.513 19.6 14.5 20.675 14.5 21.013C14.5 21.275 14.688 21.587 15.188 21.487C17.173 20.8168 18.8979 19.541 20.1199 17.8392C21.3419 16.1373 21.9994 14.0951 22 12C22 6.475 17.525 2 12 2Z" fill="#09121F"/></g><defs>
<clipPath id="clip0_8107_84105"><rect width="24" height="24" fill="white"/></clipPath></defs></svg>
"""
# Database to be used for Mongo DB
DB_NAME = "aviary"

# Name of collection in Mongo DB
COLLECTION_NAME = "event_log"

G5_COST_PER_S_IN_DOLLARS = 1.006 / 60 / 60

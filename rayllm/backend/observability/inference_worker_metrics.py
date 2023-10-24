from rayllm.backend.observability.fn_call_metrics import FnCallMetrics

prefix = "aviary_inference_worker_fn"
generate_next_token_metrics = FnCallMetrics(f"{prefix}_generate_next_token")
init_model_metrics = FnCallMetrics(f"{prefix}_init_model")
load_model_metrics = FnCallMetrics(f"{prefix}_process_new_batch")

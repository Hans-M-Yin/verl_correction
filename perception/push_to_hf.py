from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "./saved_models/0226_qwen2_5_vl_7b_mix_10K_NEW_PROMPT_210"


repo_id = "hansQAQ/0226_qwen2_5_vl_7b_mix_10K_NEW_PROMPT_210"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

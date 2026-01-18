from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "./saved_models/test_qwen2.5-vl-3b_verl_correction_data1_9k_210"

repo_id = "hansQAQ/new_qwen2.5-vl-3b_correction_9k_60step"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

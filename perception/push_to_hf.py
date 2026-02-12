from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "./saved_models/qwen2_5_vl_7b_data1_17k_NEW_CORRECTION"


repo_id = "hansQAQ/qwen2_5_vl_7b_data1_17k_NEW_CORRECTION"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

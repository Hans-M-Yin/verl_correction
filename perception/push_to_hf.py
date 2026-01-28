from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "./saved_models/qwen2_5_vl_3b_data1_6K_NEW_REWARD_220"

repo_id = "hansQAQ/qwen2_5_vl_3b_data1_6K_NEW_REWARD_220"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

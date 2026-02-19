from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "./saved_models/0219_qwen2_5_vl_3b_mix_12k_180"


repo_id = "hansQAQ/0219_qwen2_5_vl_3b_mix_12k_180"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

local_model_path = "./saved_models/0219_qwen2_5_vl_3b_mix_12k_90"


repo_id = "hansQAQ/0219_qwen2_5_vl_3b_mix_12k_90"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)


local_model_path = "./saved_models/0219_qwen2_5_vl_3b_mix_12k_249"


repo_id = "hansQAQ/0219_qwen2_5_vl_3b_mix_12k_249"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

local_model_path = "./saved_models/0219_qwen2_5_vl_3b_mix_12k_135"


repo_id = "hansQAQ/0219_qwen2_5_vl_3b_mix_12k_135"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)

import kagglehub

# 1. Log in to your Kaggle account (prompts for API token on first run)
kagglehub.login()

# 2. Direct path to your local export folder
LOCAL_MODEL_DIR = r"C:\AutoAudit\export_model"

# 3. Kaggle Model Identifiers
MODEL_SLUG = "bdsl-ctc-sign-model"
VARIATION_SLUG = "default"

# 4. Upload to Kaggle Hub under your handle 'nafitab'
kagglehub.model_upload(
    handle=f"nafitab/{MODEL_SLUG}/pytorch/{VARIATION_SLUG}",
    local_model_dir=LOCAL_MODEL_DIR,
    version_notes="Initial release of BdSL CTC Sign Model"
)
from .ssl_workaround import configure_hf_ssl_verify

# Automatically apply opt-in Hugging Face SSL verification workaround if enabled
configure_hf_ssl_verify()

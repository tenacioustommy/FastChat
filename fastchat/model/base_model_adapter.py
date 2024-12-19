
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from fastchat.model.compression import load_compress_model
from fastchat.conversation import Conversation, get_conv_template

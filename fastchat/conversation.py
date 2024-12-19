"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have any changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

import dataclasses
import os
from typing import List, Any, Dict, Union, Optional
from PIL import Image
import requests
import base64
from io import BytesIO


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    model_path: str = None
    # adapter: "BaseModelAdapter" = None
    # tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
    # processor: Any = None
    # The system message
    system_message: str = ""
    system_message_vision: str = ""
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str]] = ()
    offset: int = 2
    # The maximum image size in megabytes that this model takes in. None means we do not resize the image.
    max_image_size_mb: int = None

    # def get_prompt(self) -> str:
    #     """使用 HuggingFace tokenizer 获取对话提示。"""
    #     # self.adapter
    #     text = self.tokenizer.apply_chat_template(
    #         self.messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
        
    #     return text

    def get_images(self):
        """获取并处理消息中的图片,统一转换为 PIL Image 格式"""
        images = []
        for i, msg in enumerate(self.messages):
            # OpenAI 格式处理
            contents = msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
            for content in contents:
                if isinstance(content, dict) and content.get("type") == "image_url":
                    image_url = content["image_url"]
                    if isinstance(image_url, dict):
                        image_url = image_url["url"]
                    
                    # 处理不同格式的图片URL
                    if image_url.startswith("data:image"):
                        # Base64 格式
                        image_data = image_url.split(",")[1]
                        image = Image.open(BytesIO(base64.b64decode(image_data)))
                    else:
                        # URL 格式
                        response = requests.get(image_url)
                        image = Image.open(BytesIO(response.content))
                        
                    images.append(image)

        return images

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def get_system_message(self, is_vision=False):
        """return the system message."""
        if is_vision and self.system_message_vision:
            return self.system_message_vision
        return self.system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        from fastchat.serve.vision.image import ImageFormat

        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, images = msg
                    image = images[0]  # Only one image on gradio at one time
                    if image.image_format == ImageFormat.URL:
                        img_str = f'<img src="{image.url}" alt="user upload image" />'
                    elif image.image_format == ImageFormat.BYTES:
                        img_str = f'<img src="data:image/{image.filetype};base64,{image.base64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace("<image>\n", "").strip()

                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_vision_api_messages(self, is_mistral=False):
        """Convert the conversation to OpenAI vision api completion format"""
        if self.system_message == "":
            ret = []
        else:
            ret = [
                {
                    "role": "system",
                    "content": self.system_message,
                }
            ]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    content_list = [{"type": "text", "text": msg[0]}]
                    image_urls = msg[1]
                    for image in image_urls:
                        image_url = image.to_openai_image_format()
                        content = {}
                        if is_mistral:
                            content = {"type": "image_url", "image_url": image_url}
                        else:
                            content = {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                        content_list.append(content)

                    ret.append({"role": "user", "content": content_list})
                else:
                    ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append(
                        {
                            "role": "assistant",
                            "content": msg,
                        }
                    )
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def to_gemini_api_messages(self):
        from fastchat.utils import load_image

        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    text, images = msg[0], msg[1]
                    content_list = [text]
                    for image in images:
                        pil_image = load_image(image.base64_str)
                        content_list.append(pil_image)
                    ret.append({"role": "user", "content": content_list})
                else:
                    ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "model", "content": msg})
        return ret

    def to_vertex_api_messages(self):
        from vertexai.preview.generative_models import Image
        import base64
        import requests
        from fastchat.serve.vision.image import ImageFormat

        if self.system_message == "":
            ret = []
        else:
            ret = [self.system_message]

        for role, msg in self.messages[self.offset :]:
            if msg is not None:
                if type(msg) is tuple:
                    text, images = msg[0], msg[1]
                    for image in images:
                        if image.image_format == ImageFormat.URL:
                            response = requests.get(image.url)
                            image = response.content
                        elif image.image_format == ImageFormat.BYTES:  # base64
                            image = base64.b64decode(image.base64_str)
                        ret.append(Image.from_bytes(image))
                    ret.append(text)
                else:
                    ret.append(msg)

        return ret

    def to_anthropic_vision_api_messages(self):
        """Convert the conversation to Claude-3 Messages Vision API format"""
        ret = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            }
        ]
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    content_list = [{"type": "text", "text": msg[0]}]

                    for image in msg[1]:
                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{image.filetype}",
                                    "data": image.base64_str,
                                },
                            }
                        )

                    ret.append({"role": "user", "content": content_list})
                else:
                    ret.append(
                        {"role": "user", "content": [{"type": "text", "text": msg}]}
                    )
            else:
                if msg is not None:
                    ret.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": msg}],
                        }
                    )
        return ret

    def to_reka_api_messages(self):
        from fastchat.serve.vision.image import ImageFormat
        from reka import ChatMessage, TypedMediaContent, TypedText

        ret = []
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) == tuple:
                    text, images = msg
                    for image in images:
                        if image.image_format == ImageFormat.BYTES:
                            ret.append(
                                ChatMessage(
                                    content=[
                                        TypedText(
                                            type="text",
                                            text=text,
                                        ),
                                        TypedMediaContent(
                                            type="image_url",
                                            image_url=f"data:image/{image.filetype};base64,{image.base64_str}",
                                        ),
                                    ],
                                    role="user",
                                )
                            )
                else:
                    ret.append(
                        ChatMessage(
                            content=[
                                TypedText(
                                    type="text",
                                    text=msg,
                                )
                            ],
                            role="user",
                        )
                    )
            else:
                if msg is not None:
                    ret.append(
                        ChatMessage(
                            content=[
                                TypedText(
                                    type="text",
                                    text=msg,
                                )
                            ],
                            role="assistant",
                        )
                    )

        return ret

    def to_metagen_api_messages(self):
        """Convert the conversation to MetaGen (Meta) chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "text": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    text, images = msg[0], msg[1]
                    # Currently only support one image.
                    attachment = {
                        "type": "base64_image",
                        "mime": "image/jpeg",
                        "data": images[-1].base64_str,
                    }
                    ret.append({"role": "user", "text": text, "attachment": attachment})
                else:
                    ret.append({"role": "user", "text": msg})
            else:
                if msg is not None:
                    ret.append({"role": "ai", "text": msg})
        return ret

    def save_new_images(self, has_csam_images=False, use_remote_storage=False):
        import hashlib
        from fastchat.constants import LOGDIR
        from fastchat.utils import load_image, upload_image_file_to_gcs
        from PIL import Image

        _, last_user_message = self.messages[-2]

        if type(last_user_message) == tuple:
            text, images = last_user_message[0], last_user_message[1]

            image_directory_name = "csam_images" if has_csam_images else "serve_images"
            for image in images:
                loaded_image = load_image(image.base64_str)
                hash_str = hashlib.md5(loaded_image.tobytes()).hexdigest()
                filename = os.path.join(
                    image_directory_name,
                    f"{hash_str}.{image.filetype}",
                )

                if use_remote_storage and not has_csam_images:
                    image_url = upload_image_file_to_gcs(loaded_image, filename)
                    # NOTE(chris): If the URL were public, then we set it here so future model uses the link directly
                    # images[i] = image_url
                else:
                    filename = os.path.join(LOGDIR, filename)
                    if not os.path.isfile(filename):
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        loaded_image.save(filename)

    def extract_text_and_image_hashes_from_messages(self):
        import hashlib
        from fastchat.utils import load_image
        from fastchat.serve.vision.image import ImageFormat

        messages = []

        for role, message in self.messages:
            if type(message) is tuple:
                text, images = message[0], message[1]

                image_hashes = []
                for image in images:
                    if image.image_format == ImageFormat.URL:
                        image_hashes.append(image)
                    elif image.image_format == ImageFormat.BYTES:
                        image = load_image(image.base64_str)
                        image_hash = hashlib.md5(image.tobytes()).hexdigest()
                        image_hashes.append(image_hash)

                messages.append((role, (text, image_hashes)))
            else:
                messages.append((role, message))

        return messages

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            system_message_vision=self.system_message_vision,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            max_image_size_mb=self.max_image_size_mb,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_and_image_hashes_from_messages(),
            "offset": self.offset,
        }


def get_conv_template(model_path: str) -> Conversation:
    return Conversation(model_path=model_path)


if __name__ == "__main__":
    from fastchat.conversation import get_conv_template

    print("-- Vicuna template --")
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

    print("-- Llama-2 template --")
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a helpful, respectful and honest assistant.")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

    print("-- ChatGPT template --")
    conv = get_conv_template("chatgpt")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.to_openai_api_messages())

    print("\n")

    print("-- Claude template --")
    conv = get_conv_template("claude")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

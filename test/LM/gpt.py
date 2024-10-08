from openai import OpenAI
import base64
import requests
from PIL import Image
from io import BytesIO
import cv2
import yaml


class LargeModel():
    def __init__(self, prompt_path, config="config.yaml"):
        self.config = yaml.safe_load(open('config.yaml', 'r'))
        self.client = OpenAI(api_key=self.config["LLM"]["api_key"])
        with open(prompt_path, 'r') as f:
            self.prompt = f.read()

    def _encode_image(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def _build_messages_to_gpt(self, instruction, image_path):
        prompt = self.prompt.format(instruction=instruction)
        if image_path is not None:
            image = cv2.imread(image_path)
            img_base64 = self._encode_image(image)

        if img_base64 is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ]

        return messages

    def _download_image(self, url):
        response = requests.get(url)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            
            img.save('downloaded_image.jpg')
            print("image saved")
        else:
            print(f"error! status_code: {response.status_code}")


    def talk_to_gpt(self, instruction, image=None):
        messages = self._build_messages_to_gpt(instruction, image)

        stream = self.client.chat.completions.create(model = self.config["LLM"]["model"],
                                                messages = messages,
                                                temperature = 0.0,
                                                max_tokens = 2048,
                                                stream = True)

        gpt_answer = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                gpt_answer += chunk.choices[0].delta.content

        return gpt_answer

if __name__ == '__main__':
    gpt = LargeModel("prompt_segment.txt")
    print(gpt.talk_to_gpt("Moving forward", "gpt_input_image.jpg"))
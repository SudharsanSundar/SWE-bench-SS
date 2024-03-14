from openai import OpenAI
from chunk_utils import split_text
from embedding_utils import OpenAIEmbeddingModel
import json
import anthropic
import together

#client = OpenAI(api_key=OAI_API_KEY)
togClient = OpenAI(api_key=TOG_API_KEY, base_url='https://api.together.xyz')
#antClient = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
### input 
#together.api_key = "xxxxx"



"""
Handles calls to tgoether ai models
"""
class TogModel:
    def __init__(self,
                 model="codellama/CodeLlama-13b-Python-hf",
                 system_prompt="You are a Question Answering portal.",
                 max_tokens=1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    # def answer(self, prompt: str):
    #     completion = client.chat.completions.create(
    #         model=self.model,
    #         messages=[
    #             {"role": "system", "content": self.system_prompt},
    #             {"role": "user", "content": prompt}
    #         ]
    #     )
    #
    #     return completion

    def answer_txt(self, prompt: str) -> str:
        completion = togClient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return completion.choices[0].message.content


import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

# Bedrock imports
import boto3
import json

load_dotenv()

class OpenAIClient:
    def __init__(self, model='gpt-5-mini-2025-08-07'):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def chat(self, messages):
        if isinstance(messages, str):
            messages = [ 
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": messages}
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content


# AWS Bedrock Client
class BedrockClient:
    def __init__(self, model_id, region_name=None, anthropic_version=None):
        self.model = model_id
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.is_claude = "claude" in model_id.lower()
        self.is_cohere = "cohere" in model_id.lower()
        self.is_nova = "nova" in model_id.lower()
        self.anthropic_version = anthropic_version or "bedrock-2023-05-31"
        self.client = boto3.client(
            "bedrock-runtime", 
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    def chat(self, messages):
        # Accepts either a string or a list of dicts (OpenAI-style)
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]

        if self.is_claude:
            # Bedrock Claude models expect 'system' as top-level parameter, not a message role
            system_content = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append(msg)

            # Build request body for Claude via Bedrock
            body = {
                "anthropic_version": self.anthropic_version,
                "messages": user_messages,
                "max_tokens": 1024*8,
                "temperature": 0
            }
            
            if system_content:
                body["system"] = system_content
        elif self.is_cohere:
            # Cohere models use a different format
            # Extract the user prompt (last user message)
            user_prompt = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                    break
            
            body = {
                "message": user_prompt,
                "max_tokens": 1024*8,
                "temperature": 0
            }
        elif self.is_nova:
            # Amazon Nova models use the converse API
            # Extract system message and convert to Nova format
            system_content = None
            nova_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    nova_messages.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}]
                    })
            
            converse_args = {
                "modelId": self.model,
                "messages": nova_messages,
                "inferenceConfig": {
                    "maxTokens": 1024*8,
                    "temperature": 0
                }
            }
            
            if system_content:
                converse_args["system"] = [{"text": system_content}]
            
            response = self.client.converse(**converse_args)
            # Nova converse API returns output in response["output"]["message"]["content"][0]["text"]
            return response["output"]["message"]["content"][0]["text"]
        else:
            # For other models (OpenAI-compatible, etc), use standard format
            body = {
                "messages": messages,
                "max_tokens": 1024*8,
                "temperature": 0
            }
        
        # For non-Nova models, use invoke_model
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response["body"].read())
        
        # Claude via Bedrock returns content in 'content'[0]['text']
        if "content" in result:
            return result["content"][0]["text"]
        # Cohere returns 'text'
        if "text" in result:
            return result["text"]
        # OpenAI-style returns 'choices'[0]['message']['content']
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        # Fallback for other response formats
        return result.get("completion") or result.get("generation") or str(result)


class GeminiClient:
    def __init__(self, model='gemini-2.5-pro'):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.client = genai.GenerativeModel(model)
    
    def chat(self, messages):
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0)
        )
        return response.text


class DeepSeekClient:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model
    
    def chat(self, messages):
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content




if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "If 5 students got grades around 17-19, and another 5 around 12-13, what is the average grade?"}
    ]

    # Test Bedrock (example: Amazon Nova model)
    try:
        bedrock_model_id = "us.amazon.nova-premier-v1:0"
        bedrock_client = BedrockClient(bedrock_model_id)
        bedrock_response = bedrock_client.chat(messages)
        print("Bedrock response:", bedrock_response)
    except Exception as e:
        print("Bedrock test failed:", e)
    exit()
    # Test Bedrock Claude (with inference profile)
    try:
        bedrock_model_id = "arn:aws:bedrock:eu-north-1:805187821596:inference-profile/eu.anthropic.claude-opus-4-6-v1"
        bedrock_client = BedrockClient(bedrock_model_id, region_name="eu-north-1")
        bedrock_response = bedrock_client.chat(messages)
        print("Bedrock Claude response:", bedrock_response)
    except Exception as e:
        print("Bedrock Claude test failed:", e)


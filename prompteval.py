import argparse
import subprocess
import json
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Optional, List
from openai import OpenAI
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ModelInput(BaseModel):
    prompt: str
    user_input: Dict[str, Any]

class TestCase(BaseModel):
    ideal_answer: str
    threshold: Optional[float] = None
    user_input: Dict[str, Any]

class TestCases(BaseModel):
    test_cases: List[TestCase]

class Result(TestCase):
    response: str
    similarity: float

class Results(BaseModel):
    results: List[Result]

class Prompts(BaseModel):
    prompts: List[str] 

def parse_prompts_file(file_path: str) -> Prompts:
    with open(file_path, 'r') as file:
        prompts_data = json.loads(file.read())
    
    prompts_list = Prompts(prompts=prompts_data)
    
    return prompts_list

def parse_test_cases_file(file_path: str) -> TestCases:
    with open(file_path, 'r') as file:
        test_cases_data = json.loads(file.read())
    
    test_cases = TestCases(
        test_cases=[TestCase(**data) for data in test_cases_data]
    )
    
    return test_cases

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def calculate_similarity(text1: str, text2: str) -> float:
    embedding_text1 = get_embedding(text=text1)
    embedding_text2 = get_embedding(text=text2)

    similarity_score = 1 - cosine(embedding_text1, embedding_text2)
    
    return similarity_score

def call_model(model_path : str, payload: ModelInput):
    try:
        payload_json = payload.json() 

        results = subprocess.run(
            [
                "python",
                model_path,
                payload_json
            ],
            text=True,
            capture_output=True, 
            check=True
        )
        return results.stdout
    except ValidationError as e:
        print("validation error: ", e.json())
    except subprocess.CalledProcessError as e:
        print(f"Model script returned a non-zero exit status: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")

def validate_prompts(model, promptsDTO: Prompts, testcasesDTO: TestCases):
    resultsDTO = Results(results=[])
    for prompt in promptsDTO.prompts:
        for testcase in testcasesDTO.test_cases:
            model_output = call_model(model, ModelInput(prompt=prompt, user_input=testcase.user_input))
            similarity_score = calculate_similarity(testcase.ideal_answer, model_output)
            resultsDTO.results.append(
                Result(
                    response=model_output, 
                    ideal_answer=testcase.ideal_answer, 
                    user_input=testcase.user_input,
                    threshold=testcase.threshold,
                    similarity=similarity_score
                )
            )
    return resultsDTO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "#Argparse\nEvaluate prompts with a given model."
    )

    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--prompts', type=str, required=True, help='File path to a list of prompts')
    parser.add_argument('--testcases', type=str, required=True, help='File path to a list of test cases')
    args = parser.parse_args()

    prompts = parse_prompts_file(args.prompts)
    testcases = parse_test_cases_file(args.testcases)

    print(validate_prompts(model=args.model, promptsDTO=prompts, testcasesDTO=testcases))

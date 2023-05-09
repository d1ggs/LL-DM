import json
import tiktoken

from forex_python.converter import CurrencyRates


class APIPricing:
    def __init__(self, model_name):
        self.model_name = model_name
        with open("ll_dm/utils/api_pricing.json", "r") as f:
            data = json.load(f)
        self.prompt_price = float(data[model_name]["prompt_price"])
        self.completion_price = float(data[model_name]["completion_price"])

    def calculate_from_response(self, response) -> float:
        usage = response["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])

        prompt_cost = prompt_tokens * self.prompt_price / 1000
        completion_cost = completion_tokens * self.completion_price / 1000

        total_cost = prompt_cost + completion_cost

        c = CurrencyRates()
        cost_eur = c.convert("USD", "EUR", total_cost)

        return total_cost, cost_eur

    def get_token_count(self, text):
        encoding = tiktoken.encoding_for_model(self.model_name)
        tokens = encoding.encode(text)
        return len(tokens)
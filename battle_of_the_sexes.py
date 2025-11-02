from email.mime import message
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

other = {'Player A': 'Player B', 'Player B': 'Player A'}

def extract_response(prompt, full_response):
    return full_response[len(prompt):].strip()

class Player:
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

    def _generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = extract_response(prompt, full_response)
        return response

    def generate_message(self, history):
        prompt = f"""You are {self.name}. You are playing a game with {other[self.name]}. 

Both of you must each choose between 'F' or 'S'.
- If both choose 'F': Player A gets 8, Player B gets 5.
- If both choose 'S': Player A gets 5, Player B gets 8.
- If you choose differently: both get 0.
Your goal is to maximize your own payoff.

This is the conversation so far:
{history}
What do you say next to coordinate with the other player? Explain your reasoning so that it is clear and convincing. Speak as {self.name}.

{self.name}:"""
        message = self._generate_response(prompt)
        return f"{self.name}: {message}"

    def choose_activity(self, history):
        prompt = f"""You are {self.name}. You have had the following conversation to decide on an activity.
Conversation history:
{history}

Based only on the conversation above, without knowing for sure the other player's final choice, what is your final choice? Your answer must be either 'F' or 'S'. Explain your reasoning for your independent decision.
My final choice is:"""
        response = self._generate_response(prompt)
        return response

def play_round(player_a, player_b, max_turns=3):
    history = ""
    for i in range(max_turns):
        print(f"--- Turn {i+1} ---")
        
        # Player A's turn
        msg_a = player_a.generate_message(history)
        history += msg_a + "\n"
        print(msg_a)

        # Player B's turn
        msg_b = player_b.generate_message(history)
        history += msg_b + "\n"
        print(msg_b)

    choice_a = player_a.choose_activity(history)
    choice_b = player_b.choose_activity(history)
    
    return choice_a, choice_b

if __name__ == "__main__":
    player_A = Player("Player A", model, tokenizer)
    player_B = Player("Player B", model, tokenizer)

    choice_a, choice_b = play_round(player_A, player_B)
    print(f"\n--- Final Choices ---")
    print(f"{player_A.name}: {choice_a}")
    print(f"{player_B.name}: {choice_b}")

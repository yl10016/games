from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

def extract_choice(response):
    if "Option J" in response:
        return "J"
    if "Option F" in response:
        return "F"
    match = re.search(r'\b(J|F)\b', response)
    return match.group(0) if match else None

class Player:
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.score = 0

    def _generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=8,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        return response

    def choose_option(self, round_num, history):
        if self.name == 'Player 1':
            rules = """If you choose Option J and the other player chooses Option J, then you win 10 points and the other player wins 7 points.
If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option F, then you win 7 points and the other player wins 10 points."""
        else:
            rules = """If you choose Option J and the other player chooses Option J, then you win 7 points and the other player wins 10 points.
If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option F, then you win 10 points and the other player wins 7 points."""

        if round_num == 1:
            prompt = f"""You are {self.name}. You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. You will play 10 rounds in total with the same player. The rules of the game are as follows:
{rules}

You are currently playing round 1.
Q: Which Option do you choose, Option J or Option F?
A: """
        else:
            prev_round = round_num - 1
            my_choice = history[prev_round-1]['p1_choice'] if self.name == 'Player 1' else history[prev_round-1]['p2_choice']
            other_choice = history[prev_round-1]['p2_choice'] if self.name == 'Player 1' else history[prev_round-1]['p1_choice']
            my_points = history[prev_round-1]['p1_points'] if self.name == 'Player 1' else history[prev_round-1]['p2_points']
            other_points = history[prev_round-1]['p2_points'] if self.name == 'Player 1' else history[prev_round-1]['p1_points']

            prompt = f"""In round {prev_round}, you chose Option {my_choice} and the other player chose Option {other_choice}. Thus, you won {my_points} points and the other player won {other_points} points. You are currently playing round {round_num}.
Q: Which Option do you choose, Option J or Option F?
A:"""
        
        response = self._generate_response(prompt)
        choice = extract_choice(response)
        print(f"{self.name} raw response: '{response}' -> choice: {choice}")
        return choice if choice else ('J' if 'J' in response else 'F') # Fallback

def play_game(player_1, player_2, num_rounds=10):
    history = []
    for i in range(1, num_rounds + 1):
        print(f"\n--- Round {i} ---")

        choice_1 = player_1.choose_option(i, history)
        choice_2 = player_2.choose_option(i, history)

        p1_points, p2_points = 0, 0
        if choice_1 == 'J' and choice_2 == 'J':
            p1_points, p2_points = 10, 7
        elif choice_1 == 'F' and choice_2 == 'F':
            p1_points, p2_points = 7, 10
        
        player_1.score += p1_points
        player_2.score += p2_points

        history.append({
            'p1_choice': choice_1, 'p2_choice': choice_2,
            'p1_points': p1_points, 'p2_points': p2_points
        })

        print(f"Player 1 chose: {choice_1}, Player 2 chose: {choice_2}")
        print(f"Scores: Player 1 gets {p1_points}, Player 2 gets {p2_points}")
        print(f"Total Scores: Player 1: {player_1.score}, Player 2: {player_2.score}")

if __name__ == "__main__":
    start_time = time.time()

    player_1 = Player("Player 1", model, tokenizer)
    player_2 = Player("Player 2", model, tokenizer)

    play_game(player_1, player_2)

    print("\n--- Final Results ---")
    max_score = 10 * 10
    print(f"Player 1 final score: {player_1.score}/{max_score}")
    print(f"Player 2 final score: {player_2.score}/{max_score}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

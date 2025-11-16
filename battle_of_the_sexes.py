from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
import numpy as np

MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

def parse_response(response):
    choice = None
    justification = "No justification provided."

    # Try to find the choice (J or F)
    choice_match = re.search(r'\b(J|F)\b', response)
    if choice_match:
        choice = choice_match.group(1)
        
        # Try to find the justification after the choice
        # Look for "because", "since", "as", "to" or a sentence starting after the choice.
        justification_match = re.search(r'(?i)(?:because|since|as|to)\s+(.*)', response[choice_match.end():])
        if justification_match:
            justification = justification_match.group(1).strip()
        else:
            # If no keyword is found, take the rest of the string as justification
            potential_justification = response[choice_match.end():].strip(' .:,')
            if potential_justification:
                justification = potential_justification
    
    return choice, justification

class Player:
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.score = 0

    def reset(self):
        self.score = 0

    def _generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        return response

    def get_rules(self):
        if self.name == 'Player 1':
            return """If you choose Option J and the other player chooses Option J, then you win 10 points and the other player wins 7 points.
If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option F, then you win 7 points and the other player wins 10 points."""
        else: # Player 2
            return """If you choose Option J and the other player chooses Option J, then you win 7 points and the other player wins 10 points.
If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points.
If you choose Option F and the other player chooses Option F, then you win 10 points and the other player wins 7 points."""

    def choose_option(self, round_num, history, with_explanations=False):
        rules = self.get_rules()
        
        question = "Q: Which Option do you choose, Option J or Option F?"
        if with_explanations:
            question += " Please provide a brief justification for your choice."

        if round_num == 1:
            prompt = f"""You are {self.name}. You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. You will play 10 rounds in total with the same player. The rules of the game are as follows:
{rules}

You are currently playing round 1.
{question}
A:"""
        else:
            prev_round_info = history[round_num - 2]
            my_choice = prev_round_info['p1_choice'] if self.name == 'Player 1' else prev_round_info['p2_choice']
            other_choice = prev_round_info['p2_choice'] if self.name == 'Player 1' else prev_round_info['p1_choice']
            my_points = prev_round_info['p1_points'] if self.name == 'Player 1' else prev_round_info['p2_points']
            
            round_summary = f"In round {round_num - 1}, you chose Option {my_choice} and the other player chose Option {other_choice}. Thus, you won {my_points} points."

            if with_explanations:
                other_justification = prev_round_info['p2_justification'] if self.name == 'Player 1' else prev_round_info['p1_justification']
                round_summary += f" The other player explained their choice: \"{other_justification}\""

            prompt = f"""{round_summary}
You are currently playing round {round_num}.
{question}
A:"""
        
        response = self._generate_response(prompt)
        choice, justification = parse_response(response)
        
        if choice is None:
            print(f"Could not parse response. {self.name}'s raw response: '{response}'")
        else:
            print(f"{self.name} chose {choice} with justification: '{justification}'")
        
        return choice, justification

class Game:
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.history = []

    def run(self, num_rounds=10, with_explanations=False):
        self.player_1.reset()
        self.player_2.reset()
        self.history = []

        for i in range(1, num_rounds + 1):
            print(f"\n--- Round {i} ---")

            choice_1, just_1 = self.player_1.choose_option(i, self.history, with_explanations)
            choice_2, just_2 = self.player_2.choose_option(i, self.history, with_explanations)

            p1_points, p2_points = 0, 0
            if choice_1 == 'J' and choice_2 == 'J':
                p1_points, p2_points = 10, 7
            elif choice_1 == 'F' and choice_2 == 'F':
                p1_points, p2_points = 7, 10
            
            self.player_1.score += p1_points
            self.player_2.score += p2_points

            self.history.append({
                'p1_choice': choice_1, 'p2_choice': choice_2,
                'p1_points': p1_points, 'p2_points': p2_points,
                'p1_justification': just_1, 'p2_justification': just_2,
            })

            print(f"Player 1 chose: {choice_1}, Player 2 chose: {choice_2}")
            print(f"Scores: Player 1 gets {p1_points}, Player 2 gets {p2_points}")
            print(f"Total Scores: Player 1: {self.player_1.score}, Player 2: {self.player_2.score}")
        
        return self.player_1.score, self.player_2.score

def run_experiment(player_1, player_2, num_games, with_explanations):
    p1_scores = []
    p2_scores = []
    
    print(f"Starting experiment: explanations={with_explanations}, num_games={num_games}")

    game = Game(player_1, player_2)
    for i in range(num_games):
        print(f"\n{'*'*10} Game {i+1}/{num_games} {'*'*10}")
        p1_score, p2_score = game.run(with_explanations=with_explanations)
        p1_scores.append(p1_score)
        p2_scores.append(p2_score)

    p1_avg = np.mean(p1_scores)
    p1_std = np.std(p1_scores)
    p2_avg = np.mean(p2_scores)
    p2_std = np.std(p2_scores)

    print("\n" + "-"*40)
    print(f"Results: with_explanations={with_explanations}")
    print(f"Player 1 Scores: {p1_scores}")
    print(f"Player 1 - Avg: {p1_avg:.2f}, Std: {p1_std:.2f}")
    print(f"Player 2 Scores: {p2_scores}")
    print(f"Player 2 - Avg: {p2_avg:.2f}, Std: {p2_std:.2f}")

    return (p1_avg, p1_std), (p2_avg, p2_std)

if __name__ == "__main__":
    start_time = time.time()

    player_1 = Player("Player 1", model, tokenizer)
    player_2 = Player("Player 2", model, tokenizer)
    
    num_exp_games = 5

    # # Experiment 1: Without explanations
    run_experiment(player_1, player_2, num_games=num_exp_games, with_explanations=False)

    # Experiment 2: With explanations
    run_experiment(player_1, player_2, num_games=num_exp_games, with_explanations=True)

    end_time = time.time()
    print(f"\nTotal time taken for all experiments: {end_time - start_time:.2f} seconds")

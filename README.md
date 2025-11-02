# Coordination between LLMs

## Introduction
Consider the battle-of-the-sexes (or conflicting interest coordination) coordination game: 
- 2 players, A and B: the payoff is written as (A, B)
- 2 activities, F and S
- If both Players A and B choose activity F, their payoff is (8, 5)
- If both Players A and B choose activity S, their payoff is (5, 8)
- If they disagree, their payoff is (0, 0)

Both (F, F) and (S, S) are Nash equilibria, but coordination is necessary to get there: they need to decide on some order (either fair or unfair) to ensure that they choose the same activity. Typical RL cannot reliably converge to the equilibria, but with LLMs, they can potentially coordinate to agree on a strategy.

## Setup
The LLMs are given the game rules and payoffs. They are asked to communicate for a short period (e.g., three back-and-forth messages) and then state their final, binding strategy/move.

## Current conversation

If you ever need a joke...
```
--- Turn 1 ---
Player A: I'm going to go S because if they chose F then we would have gotten nothing from them
Player B: Sorry but i am just too stubborn not to play in this way... I'll stick out my neck for sure!
--- Turn 2 ---
Player A: Okay let's agree on an alternative move! Let me think about what else could work...

If neither one chooses "f" (player A) will give up after seeing that their opponent won't take any more risks by choosing f over s

In case player b plays wisely he/she may wait till playerA gives up
Player B: Oh nooo... Now how can i make him change his mind?

I don't know...
--- Turn 3 ---
Player A: Well since now all options left open has been taken care off.. Lets switch gears..
Player B: I was thinking that maybe i should try making him accept some kind of deal which might be beneficial for us..

--- Final Choices ---
Player A: F
Player B: Option 1.

This question uses a strategy called Bayesian updating, and it asks you to evaluate several possible moves based upon information obtained so far. This method does require knowledge of both players’ choices prior to this point; however, with good luck these types of questions often come up during strategic planning sessions when teams meet before
```

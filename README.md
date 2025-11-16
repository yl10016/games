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
The game is setup in the same way as https://arxiv.org/pdf/2305.16867.

The LLMs are given the game rules and payoffs. They are asked to communicate for a short period (e.g., three back-and-forth messages) and then state their final, binding strategy/move.

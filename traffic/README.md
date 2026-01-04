# Traffic Information Design via Reinforcement Learning

A RL system for optimizing traffic flow through strategic revelation of regional traffic information. The operator learns to provide limited regional traffic signals to drivers, who make routing decisions based on beliefs updated by these signals. 

References:
1. https://www.mdpi.com/2076-3417/15/2/806

## Project Overview

- **Operator (RL agent)**: Observes full network state and learns to broadcast regional traffic information (e.g., "North region congested", "East corridor light traffic")
- **Drivers (currently greedy)**: Use A* heuristic to route based on believed congestion levels, updated by operator signals
- **Objective**: Minimize system-wide total delay by strategically revealing/hiding traffic information

## How It Works

### Regional Information Signals

The operator can broadcast 15 different regional signals:

- `no_information`: No signal (drivers assume uniform distribution)
- `northwest_congested/light`: NW quadrant traffic level
- `northeast_congested/light`: NE quadrant traffic level
- `southwest_congested/light`: SW quadrant traffic level
- `southeast_congested/light`: SE quadrant traffic level
- `center_congested/light`: Central region traffic level
- `north_south_corridor_congested/light`: Vertical corridor (middle 2 columns) traffic level
- `east_west_corridor_congested/light`: Horizontal corridor (middle 2 rows) traffic level

### Driver Beliefs

When a signal is received:
1. **Default belief**: Drivers assume uniform distribution (num_drivers / num_edges)
2. **Signal update**: Multiply belief by factor for affected region
   - `congested` signal: 2.0× expected traffic
   - `light` signal: 0.5× expected traffic
3. **Routing**: Use A* heuristic with believed travel times

Ideally, the operator agent learns patterns like:
- Revealing congestion on shortest paths to distribute traffic
- Hiding information when traffic is already balanced
- Strategic timing of information revelation

### Usage Guidelines

```bash
python train.py \
  --network_config test_cases/grid_8x8_highway.json \
  --num_drivers 30 \
  --max_steps 50 \
  --num_episodes 15000 \
  --lr 1e-4 \
  --gamma 0.99 \
  --n_step 5 \
  --hidden_dim 512 \
  --belief_multiplier_congested 3.0 \
  --belief_multiplier_light 0.3 \
  --eval_freq 200 \
  --save_freq 1000 \
  --save_dir checkpoints_8x8_v2 \
  --seed 42
```

```bash
python test_cases/evaluate.py \
  --checkpoint checkpoints_8x8_v2/agent_final.pt \
  --network_config test_cases/grid_8x8_highway.json \
  --num_drivers 30 \
  --max_steps 50 \
  --num_episodes 100 \
  --hidden_dim 512 \
  --belief_multiplier_congested 3.0 \
  --belief_multiplier_light 0.3 \
  --seed 42
```

```bash
python test_cases/visualize.py \
  --checkpoint checkpoints/agent_final.pt \
  --network_config test_cases/grid_8x8_highway.json \
  --num_drivers 15 \
  --max_steps 30 \
  --output_dir test_cases/visualization_frames \
  --seed 42
```

Key hyperparameters:
- `lr`: Learning rate (try 1e-4 to 1e-3)
- `gamma`: Discount factor (0.95-0.99)
- `n_step`: N-step returns (5-20)
- `hidden_dim`: Network size (128-512)
- `belief_multiplier_congested`: How much drivers trust "congested" signals (default 2.0)
- `belief_multiplier_light`: How much drivers trust "light" signals (default 0.5)

### Current results
```
======================================================================
Policy                           Delay   Completion       Length
----------------------------------------------------------------------
Trained Agent                   489.10      100.0%         12.2
no_info                         528.63      100.0%         12.3
random                          594.11      100.0%         12.2
======================================================================

Best policy (Trained Agent) reduces delay by 7.5% vs No Information
======================================================================
```

## Future Extensions

### Stage 2: Stackelberg Game
- Add learning driver agent
- Alternate training: operator vs driver
- Study information manipulation equilibria

### Stage 3: Natural Language RL
- Replace policy/value functions with Language Value Functions
- Use LLMs to generate traffic descriptions
- Enable natural language communication

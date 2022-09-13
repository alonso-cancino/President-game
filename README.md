# Reinforcement learning model for playing the president card game.

## Rules of the game:

The game is played with a $40$ card spanish deck (this means removing all $8$ s and $9$ s from a usual 58 card deck), the game can be played by $2, 4, 5$ or $8$ players (numbers which can wholly divide the deck), the goal of the game is dumping your hand.

<p align="center">
  <img width="600" height="400" src="https://user-images.githubusercontent.com/62409116/189929379-18174679-c6ec-4e38-8f1a-239bfa527efb.PNG">
</p>

At each turn, a player has 2 options, either they can pass the turn, or they can play a set of cards which are of the same number and that:
1. If the player doesn't start the round, the cards have to be the same amount of cards previously played and of strictly greater value,
2. If the player starts the round, any set of cards which are of the same number suffice.
A player starts the round if they have the $2$ of golds and it's the first round of the game, if everyone else passed the set of cards they played, or if the player before them won the game and everyone passes their play.

The value of the cards is given by:
$$1>3>12>11>10>7>6>5>4>2$$
There's only one special card which is the $2$ of golds, which is the card that starts the first round of the game (this player can still start the game with more than one card numbered $2$).

<p align="center">
  <img width="60" height="85" src="https://user-images.githubusercontent.com/62409116/189929907-e0884b09-df79-49d5-bf6e-e24ea1028062.JPG">
</p>

The first player to be without cards in hand in a game is named the president, the last player is named the servant. On each game, the servant gives the president its best card, and the president returns whatever card they don't want (different types of in-game economies can be defined instead).

## Program architecture:

There's 3 main files:

1. president_game.py: Implements the president_game class which encodes the game logic, with several functions and attributes.
  - game.reset(): Deals a hand to every player, and sets the starting player as the player which has the $2$ of golds in hand.
  - game.possible_plays(): Displays all the possible plays of the current player (game.active_player), if there's no valid plays from the active player other than passing, the function passes the turn to the next player (this computes a powerset of a set of size 20 when there's 2 players, so it can a bit computationally expensive).
  - game.play_select(k): Selects the k-th numbered play retrieved from game.possible_plays(), and advances the game state after making such play. 
  - game.players_data: Stores info on the game state, this is each players hand, cards on the stack, cards already played, active player and last player.
  - game.get_torch_state(): Gets the torch state seen if the agent was the active player (this is hand lengths of opponents, cards in the stack, cards already played, and cards in hand).
3. agent_training.py: Defines encoder and decoder functions that translates game states into torch tensors for training, then defines the neural network used for deep Q-learning, defines the masking function that hides illegal actions from the agent, and sets the training loop saving the current neural network values, stores the trained model weights into the file model_weights_ .pt .
5. test_agent.py: Tests the trained model exploiting the trained weights, for a fixed number of games against random agents.




## Program architecture:

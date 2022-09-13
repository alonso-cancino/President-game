# Reinforcement learning model for playing the "president" card game.

Rules of the game:
The game is played with a $40$ card spanish deck (this means removing all $8$ s and $9$ s from a usual 58 card deck), the game can be played by $2, 4, 5$ or $8$ players (numbers which can wholly divide the deck), the goal of the game is dumping your hand, the first player to be without cards in hand is named the president, the last player is named the servant. 
At each turn, a player has 2 options, either they can pass the turn, or they can play a set of cards which are of the same number and that:
1. If the player doesn't start the round, the cards have to be the same amount of cards previously played and of strictly greater value,
2. If the player starts the round, any set of cards which are of the same number suffice.
A player starts the round if they have the $2$ of golds and it's the first round of the game, or if everyone else passed the set of cards they played, or if the player before them won the game and everyone passes their play,
the value of the cards is given by:
$$1>3>12>11>10>7>6>5>4>2$$
There's only one special card which is the $2$ of golds, which is the card that starts the first round of the game (this player can still start the game with more than one card numbered $2$).

On each game, the servant gives the president its best card, and the president returns whatever card they don't want (different types of economy can be defined instead).

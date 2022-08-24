import tkinter as tk
from PIL import ImageTk, Image
from president_game import *
from time import time


deck = []
for i in range(10):
    for p in ('E','O','C','B'):
        deck.append(str(i)+p)
        
N_PLAYERS = int(input('Number of players: '))

root = tk.Tk()
root.geometry('800x600')

global game
    
game = president_game(N_PLAYERS)
game.reset()

while game.active_player != 0:
    pos_plays = game.possible_plays()
    selected_play = random.randint(0,len(pos_plays)-1)
    game.play_select(selected_play)

deck_imgs = []
for card in deck:
    deck_imgs.append(ImageTk.PhotoImage(Image.open(f'img/{card}.JPG').resize((60,80))))
    
cardback_img = ImageTk.PhotoImage(Image.open('img/cardback.JPG').resize((60,80)))

def display_stack():
    if game.stack != ['*'] and game.stack != []:
        for index,card in enumerate(game.stack):
            stack_labels[index] = deck_imgs[deck.index(card)]
            
            
def display_enemy_hand_len():
    hand_len = [len(game.players_data[agent]) for agent in range(N_PLAYERS) if agent!=0]
    return hand_len
    
def reset():
    game.reset()
    return

option_board = tk.Frame(root)
option_board.grid(row=1,column=1)

btn = tk.Button(option_board,text='Reiniciar partida',font='Helvetica 12',command=reset)
btn.pack(padx=200,pady=50)

player_hand = tk.Frame(root,bg='red')
player_hand.grid(row=4,column=1)

player_hand_labels = []
for ind,card in enumerate(game.players_data[0]):
    player_hand_labels.append(tk.Label(player_hand,image = deck_imgs[deck.index(card)]))
    player_hand_labels[-1].grid(row=1,column=ind)

opponents_hands = tk.Frame(root)
opponents_hands.grid(row=2,column=1)


lens = display_enemy_hand_len()
opp_hand_labels = []
opp_hand_images = []
for ind,L in enumerate(lens):
    opp_hand_labels.append(tk.Label(opponents_hands,text=f'P{ind+1} has '+str(L)+' cards in hand.'))
    opp_hand_images.append(tk.Label(opponents_hands,image = cardback_img))
    opp_hand_labels[-1].grid(row=2,column=ind)
    opp_hand_images[-1].grid(row=1,column=ind)


stack = tk.Frame(root)
stack.grid(row=3,column=1)

stack_labels = []
for ind, card in enumerate(game.stack):
    stack_labels.append(tk.Label(stack,image = deck_imgs[deck.index(card)]))
    stack_labels[-1].grid(row=1,column=ind)


root.mainloop()

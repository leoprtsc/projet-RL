# V2  tic tac toe
################################################################################################

import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Plateau:
    def __init__(self):
        self.plato_base = np.zeros((3, 3))
        self.plato = self.plato_base.copy()
        self.actionsd = {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (2, 0),
            8: (2, 1),
            9: (2, 2),
        }

    def reset(self):
        self.plato = self.plato_base.copy()
        self.nx_pl = self.plato_base.copy()

    def get_actions(self, pl):
        ad = []
        for i in range(1, 10):
            if pl[self.actionsd[i]] == 0:
                ad.append(i)
        return ad

    def jouer_coup(self, act, j_num):
        self.nx_pl = self.plato.copy()
        self.nx_pl[self.actionsd[int(act)]] = j_num
        return self.nx_pl

    def check_victoire(self, pl):
        for l in range(3):
            if np.prod(pl[l, :]) == 1 or np.prod(pl[l, :]) == 8:
                return True
        for c in range(3):
            if np.prod(pl[:, c]) == 1 or np.prod(pl[:, c]) == 8:
                return True
        if (
            np.prod([pl[(j, j)] for j in range(3)]) == 1
            or np.prod([pl[(j, j)] for j in range(3)]) == 8
        ):
            return True
        if (
            np.prod([pl[(j, 2 - j)] for j in range(3)]) == 1
            or np.prod([pl[(j, 2 - j)] for j in range(3)]) == 8
        ):
            return True
        return False

    def get_etat(self):
        return "".join(map(str, self.plato.flatten()))

    def display(self):
        print(self.plato)


def creer_modele():
    """créer les resaux de neurones pour les IA"""
    model = Sequential()
    model.add(Dense(30, input_shape=(9,), activation="linear"))
    model.add(Dense(30, activation="linear"))
    model.add(Dense(9, activation="linear"))
    model.compile(loss="mse", optimizer=Adam())
    return model


class Player:
    """Classe représentant les joueurs et ses stats et ses modèle pour les instances d'IA"""

    def __init__(self, is_human, trainable=True, val_jeton=1):
        self.is_human = is_human
        self.memory = []
        self.win_nb = 0
        self.lose_nb = 0
        self.trainable = trainable
        self.batch_size = 64
        self.epoch = 10
        self.gamma = 0.9
        self.model = creer_modele()
        self.val_jeton = val_jeton
        self.epsilon = 0.9

    def reset_stat(self):
        self.win_nb = 0
        self.lose_nb = 0

    def opti_action(self):
        """sélectionnera l'action optimal du modèle selon l'état"""
        Q = self.model.predict(np.array([game.plato.flatten()]))[0]
        Q = np.where(
            game.plato.flatten() != 0, -np.inf, Q
        )  # Ne sélectionner que les actions valides
        action = np.argmax(Q) + 1
        return action

    def play(self, game):
        actions = game.get_actions(
            game.plato
        )  # défini les actions dispos pour le choix aléatoire

        if self.is_human is False:
            # si le nb aléatoire est inf a epsi alors action aléatoire sinn action opti
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(actions)

            else:  # Or greedy action
                action = self.opti_action()
        else:
            action = int(
                input("à vous de jouez, voici les actions autorisées : " + str(actions))
            )

        return action

    def add_transition(self, n_tuple):
        # Ajoute la transition à la mémoire: tuple (s, a , r, s')
        self.memory.append(n_tuple)
        s, a, r, sp = n_tuple

    def train(self):
        """permet l'entrainement du RN"""
        if not self.trainable or self.is_human is True:
            return
        # Entraîner le modèle du réseau de neurones sur un échantillon de la mémoire
        if len(self.memory) >= self.batch_size:
            mini_batch = np.random.choice(
                len(self.memory), size=self.batch_size, replace=False
            )
            X = np.array([self.memory[i][0] for i in mini_batch])
            y = self.model.predict(X)

            self.mini_memory = [self.memory[i] for i in mini_batch]

            for i, (etat, action, recompense, etat_suivant) in enumerate(
                self.mini_memory
            ):
                if etat_suivant is None:
                    z = 0
                elif recompense == -1:
                    z = 0
                else:
                    z = np.max(self.model.predict(np.array([etat_suivant]))[0])
                y[i][action - 1] = recompense + self.gamma * z
            self.model.fit(X, y, epochs=self.epoch, verbose=0)


def train_play(game, p1, p2, p1_train=True, p2_train=True, ajout_exp_mem=True):
    game.reset()
    players = [p1, p2]
    random.shuffle(players)
    p = 0

    while (
        game.check_victoire(game.plato) is False
        and len(game.get_actions(game.nx_pl)) > 0
    ):
        if players[p % 2].is_human:
            game.display()

        action = players[p % 2].play(game)

        game.jouer_coup(action, players[p % 2].val_jeton)

        if game.check_victoire(game.nx_pl):
            reward = 1  # si le coup permet de gagner, la reward du joueur sera plus 1
        else:
            reward = 0  # si le coup ne permet pas de gagner

        if reward != 0:
            # Update stat of the current player
            players[(p + 1) % 2].lose_nb += 1.0
            players[p % 2].win_nb += 1.0

        if ajout_exp_mem:
            players[p % 2].add_transition((game.plato.flatten(), action, reward, None))

            if p != 0:
                s, a, r, sp = players[(p + 1) % 2].memory[-1]
                players[(p + 1) % 2].memory[-1] = (
                    s,
                    a,
                    reward * -1,
                    game.nx_pl.flatten(),
                )  ### permet surtout  de mettre la reward a -1 pour le coup qui a mené le joueur suivant a pv gg

        game.plato = game.nx_pl.copy()

        p += 1
        if p1_train:
            p1.train()
        if p2_train:
            p2.train()


game = Plateau()  # création du jeu
human1 = Player(
    is_human=True, trainable=False, val_jeton=1
)  # création du joueur humain jouant les 1
human2 = Player(
    is_human=True, trainable=False, val_jeton=2
)  # création du joueur humain jouant les 2
random_player1 = Player(
    is_human=False, trainable=False, val_jeton=1
)  # création du joueur aléatoire jouant les 1
random_player2 = Player(
    is_human=False, trainable=False, val_jeton=2
)  # création du joueur aléatoire jouant les 2
ia_player1 = Player(
    is_human=False, trainable=True, val_jeton=1
)  # création du joueur ia jouant les 1
ia_player2 = Player(
    is_human=False, trainable=True, val_jeton=2
)  # création du joueur ia jouant les 2

# train_play(game, human1, random_player2,False,False)


def loadmem():
    path3 = "D:/projet stage/V2/"
    ia_player1.model = tf.keras.models.load_model(path3 + "mod_ia_p1v2.h5")
    ia_player2.model = tf.keras.models.load_model(path3 + "mod_ia_p2v2.h5")
    with open(
        path3+"ia_player1.memoryv2.pickle", "rb"
    ) as fichier:
        ia_player1.memory = pickle.load(fichier)
    with open(
        path3+"ia_player2.memoryv2.pickle", "rb"
    ) as fichier:
        ia_player2.memory = pickle.load(fichier)
    with open(
        path3+"ia_player1.epsilonv2.pickle", "rb"
    ) as fichier:
        ia_player1.epsilon = pickle.load(fichier)
    with open(
        path3+"ia_player2.epsilonv2.pickle", "rb"
    ) as fichier:
        ia_player2.epsilon = pickle.load(fichier)


#loadmem() si besoins de load les élements déja enregistrer


#### Pour changer le nombre de partie 


for i in range(0, 100):
    print("partie en cours :", i)
    if i % 5 == 0:
        ia_player1.epsilon = max(ia_player1.epsilon * 0.996, 0.05)
        ia_player2.epsilon = max(ia_player2.epsilon * 0.996, 0.05)
    train_play(game, ia_player1, ia_player2, p1_train=True, p2_train=True)


######### LANCER LE CODE JUSQUE ICI

### Pour enregister les modeles et élements importants

### pickle
import pickle


path3 = "D:/projet stage/V2/"
ia_player1.model.save(path3 + "mod_ia_p1v2.h5")
ia_player2.model.save(path3 + "mod_ia_p2v2.h5")

with open(
    path3+"ia_player1.memoryv2.pickle", "wb"
) as fichier:
    pickle.dump(ia_player1.memory, fichier)
with open(
    path3+"ia_player2.memoryv2.pickle", "wb"
) as fichier:
    pickle.dump(ia_player2.memory, fichier)
with open(
    path3+"ia_player1.epsilonv2.pickle", "wb"
) as fichier:
    pickle.dump(ia_player1.epsilon, fichier)
with open(
    path3+"ia_player2.epsilonv2.pickle", "wb"
) as fichier:
    pickle.dump(ia_player2.epsilon, fichier)


for i in range(30):
    print(i)
    ia_player1.train()
    ia_player2.train()


len(ia_player1.memory)
len(ia_player2.memory)
ia_player1.epsilon
ia_player2.epsilon


# test pour voir le nb de victoire de l'ia sur le random

ia_player1.win_nb = 0
random_player2.win_nb = 0
for _ in range(1000):
    train_play(
        game,
        ia_player1,
        random_player2,
        p1_train=False,
        p2_train=False,
        ajout_exp_mem=False,
    )
random_player2.win_nb
ia_player1.win_nb

# humain contre ia 

train_play(game, human1, ia_player2, p1_train=False, p2_train=False)
train_play(game, human2, ia_player1, p1_train=False, p2_train=False)




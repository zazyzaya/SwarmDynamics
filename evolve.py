from argparse import ArgumentParser
from time import time

from joblib import Parallel, delayed
from tqdm import tqdm
import torch

from src.dna import GenePool
from src.env import Env

MAX_GAME_LEN = 5000
POPULATION = 10_000
GAME_SIZE = 100
DEVICE = 'cpu' # Why is this faster than my GPU??

def generation(gene_pool: GenePool):
    st = time()
    lineup = torch.randperm(gene_pool.population)
    n_games = gene_pool.population // (GAME_SIZE*2)

    print("\tSimulating... ", end='', flush=True)
    def game(g):
        winners = []

        blue = lineup[g*GAME_SIZE : (g+1)*GAME_SIZE]
        red = lineup[(g+1)*GAME_SIZE : (g+2)*GAME_SIZE]

        env = Env(*gene_pool.phenotype(blue), *gene_pool.phenotype(red))
        game_over = torch.zeros(2)

        for _ in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        draw = game_over.sum() == 0 or game_over.sum() == 2
        b_s, b_k, r_s, r_k = env.get_stats(top_k=GAME_SIZE//10)

        # Top half of winners from both teams
        if draw:
            n = GAME_SIZE//10
            winners.append(blue[b_s.index[:n]])
            winners.append(blue[b_k.index[:n]])
            winners.append(red[r_s.index[:n]])
            winners.append(red[r_k.index[:n]])

        # Otherwise, just the winning team survives
        else:
            team = blue if game_over[1] else red
            killers = b_k if game_over[1] else r_k
            survivors = b_s if game_over[1] else r_s

            winners.append(team[killers.index])
            winners.append(team[survivors.index])

        return torch.cat(winners)

    winners = Parallel(n_games, prefer='processes')(
        delayed(game)(g) for g in range(0, n_games*2, 2)
    )
    en = time()

    print(f" ({en-st:0.2f}s)")

    winners = torch.cat(winners)
    gene_pool.reproduce(winners)


def evaluate(gene_pool: GenePool):
    st = time()
    default = GenePool(GAME_SIZE, device=DEVICE, use_baseline=True)

    lineup = torch.randperm(gene_pool.population)
    n_games = gene_pool.population // GAME_SIZE

    print("\tEvaluating... ", end='', flush=True)
    def game(g):
        blue = lineup[g*GAME_SIZE : (g+1)*GAME_SIZE]

        env = Env(*gene_pool.phenotype(blue), *default.phenotype())
        game_over = torch.zeros(2)

        for step in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        # Evolved swarm lost or drew
        if game_over[0]:
            return MAX_GAME_LEN
        else:
            return step

    steps_to_win = Parallel(n_games, prefer='processes')(
        delayed(game)(g) for g in range(0, n_games)
    )
    en = time()
    print(f'({en-st:0.2f}s)')

    return steps_to_win

def train():
    pool = GenePool(POPULATION, device=DEVICE)

    log = 'log.txt'
    best = MAX_GAME_LEN

    for e in range(1,100_000):
        generation(pool)

        scores = evaluate(pool)
        avg = MAX_GAME_LEN - (sum(scores) / len(scores))
        max_v = MAX_GAME_LEN - min(scores)

        print(f'[{e}] Avg: {int(avg)}, Best: {max_v}', end='')
        if avg > best:
            best = avg
            print('*')
            pool.save('genes/best.pt')
        else:
            print()

        if e % 100 == 0:
            pool.save(f'genes/{e // 100}.pt')

        with open(log, 'a') as f:
            f.write(f'{e},{avg},{max_v}\n')

        pool.save('genes/current.pt')

if __name__ == '__main__':
    with open('log.txt', 'w+') as f:
        f.write('generation,avg,best\n')
    train()
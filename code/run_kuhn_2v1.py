import argparse
from pokertrees import *
from pokergames import *
from pokerstrategy import *
from card import Card
import numpy as np


def parse_list(input_str):
    return input_str.split(',')

def sum_dict(dict1, dict2, weights):
    res = {}
    for key in dict1.keys():
            res[key] = [dict1[key][i] * weights[0] + dict2[key][i] * weights[1] if dict1[key][i] != None else None for i in range(len(dict1[key]))]
    return res

def policy_gradient_descent(strategy, grad, lr):
    for infoset in grad.keys():
        for action in range(len(grad[infoset])):
            if grad[infoset][action] == None:
                illegal = action
            else:
                strategy.policy[infoset][action] -= lr * grad[infoset][action]
        prob = [strategy.policy[infoset][a] for a in range(len(strategy.policy[infoset])) if a != illegal]
        prob = L2proj_simplex(prob)
        j = 0
        for a in range(len(strategy.policy[infoset])):
            if a != illegal:
                strategy.policy[infoset][a] = prob[j]
                j += 1


if __name__ == "__main__":
    """
    3-player Kuhn Poker ([0,2] vs [1])
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4, help="rank of the deck")
    parser.add_argument("--maxbets", type=int, default=1, help="max number of bets")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed of initial strategy")
    parser.add_argument("--n_iter", type=int, default=500, help="number of iterations")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--algo", type=parse_list, default=["ND", "IPG-BR", "IBR-BR", "Cycle BR"], help="algorithms to run")
    parser.add_argument("--save_path_prefix", type=str, default="../results/2v1Kuhn", help="save path prefix")
    args = parser.parse_args()
    
    players = 3
    rank = args.rank
    maxbets = args.maxbets
    random_seed = args.random_seed

    global_n_iter = args.n_iter
    lr = args.lr

    assert rank > 3 and rank < 14
    deck = [Card(14-r,1) for r in range(rank)]
    betsize = 1 
    rounds = [RoundInfo(holecards=1,boardcards=0,betsize=betsize,maxbets=[maxbets,maxbets,maxbets])]
    ante = 1 
    blinds = None 
    gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format) 
    teams=[[0,2],[1]]
    gametree = GameTree(gamerules, teams)
    gametree.build()

    """
    available algorithms:
    ND
    IPG-BR
    IBR-BR
    Cycle BR
    """
    algo_list = args.algo

    save_path = args.save_path_prefix + "/r{}m{}lr{}/seed{}".format(rank, maxbets, lr, random_seed)

    last_wc_values = {k : None for k in algo_list}

    depth = 0
    for s in gametree.information_sets:
        depth = max(depth, len(s))
    depth -= 3
        
    print("running 2v1Kuhn_r{}m{}: #terminal={}, #infoset={}, depth={}".format(rank, maxbets, gametree.num_terminal_nodes, len(gametree.information_sets), depth))
    print("algorithms: {}".format(algo_list))
    print("hyperparameters: n_iter={}, lr={}, random_seed={}".format(global_n_iter, lr, random_seed))

    if "ND" in algo_list:
        s0 = Strategy(0)
        s1 = Strategy(1)
        s2 = Strategy(2)
        if os.path.exists(save_path):
            s0.load_from_file(save_path + "/ND/p0.strat")
            s1.load_from_file(save_path + "/ND/p1.strat")
            s2.load_from_file(save_path + "/ND/p2.strat")
            nashconv_list = list(np.load(save_path + "/ND/nashconvs.npy"))
            if len(nashconv_list) >= global_n_iter:
                n_iter = 0
                print(f"nashconv descent already satisfies global_n_iter: {len(nashconv_list)} >= {global_n_iter}\n")
            else:
                n_iter = global_n_iter - len(nashconv_list)
                print(f"continue nashconv descent, current length is {len(nashconv_list)}, {n_iter} iterations left\n")
        else:
            print(f"start nashconv descent, {global_n_iter} iterations left\n")
            s0.build_random(gametree, random_seed)
            s1.build_random(gametree, random_seed)
            s2.build_random(gametree, random_seed)
            nashconv_list = []
            n_iter = global_n_iter
        profile = StrategyProfile(gamerules, [s0,s1,s2], teams)

        for n in range(1, 1+n_iter):
            # best response
            br, ev_br = profile.best_response(br_players=[0,1,2])
            profile_br2 = StrategyProfile(gamerules, [profile.strategies[0], profile.strategies[1], br.strategies[2]], teams, gradient_only=True)
            profile_br1 = StrategyProfile(gamerules, [profile.strategies[0], br.strategies[1], profile.strategies[2]], teams, gradient_only=True)
            profile_br0 = StrategyProfile(gamerules, [br.strategies[0], profile.strategies[1], profile.strategies[2]], teams, gradient_only=True)

            # compute nashconv
            nashconv = ev_br[0] + ev_br[1] + ev_br[2]
            nashconv_list.append(nashconv)

            # update player 0
            grad1, _ = profile_br2.best_response(ga_players=[0])
            grad2, _ = profile_br1.best_response(ga_players=[0])
            total_grad = sum_dict(grad1.strategies[0].policy, grad2.strategies[0].policy, weights=[1,-2])
            policy_gradient_descent(profile.strategies[0], total_grad, lr)
            # update player 2
            grad1, _ = profile_br0.best_response(ga_players=[2])
            grad2, _ = profile_br1.best_response(ga_players=[2])
            total_grad = sum_dict(grad1.strategies[2].policy, grad2.strategies[2].policy, weights=[1,-2])
            policy_gradient_descent(profile.strategies[2], total_grad, lr)
            # update player 1
            grad1, _ = profile_br2.best_response(ga_players=[1])
            grad2, _ = profile_br0.best_response(ga_players=[1])
            total_grad = sum_dict(grad1.strategies[1].policy, grad2.strategies[1].policy, weights=[-0.5,-0.5])
            policy_gradient_descent(profile.strategies[1], total_grad, lr)

            if n % 500 == 0:
                print(f"2v1Kuhn_r{rank}m{maxbets}seed{random_seed} ND checkpoint {n}: nashconv={nashconv_list[-1]}")
                profile.strategies[0].save_to_file(save_path + "/ND/p0.strat")
                profile.strategies[1].save_to_file(save_path + "/ND/p1.strat")
                profile.strategies[2].save_to_file(save_path + "/ND/p2.strat")
                np.save(save_path + "/ND/nashconvs.npy", np.array(nashconv_list))

        profile.strategies[0].save_to_file(save_path + "/ND/p0.strat")
        profile.strategies[1].save_to_file(save_path + "/ND/p1.strat")
        profile.strategies[2].save_to_file(save_path + "/ND/p2.strat")
        np.save(save_path + "/ND/nashconvs.npy", np.array(nashconv_list))

    if "IPG-BR" in algo_list:
        s0 = Strategy(0)
        s1 = Strategy(1)
        s2 = Strategy(2)
        if os.path.exists(save_path + "/IPG-BR"):
            s0.load_from_file(save_path + "/IPG-BR" + "/p0.strat")
            s1.load_from_file(save_path + "/IPG-BR" + "/p1.strat")
            s2.load_from_file(save_path + "/IPG-BR" + "/p2.strat")
            nashconvs = list(np.load(save_path + "/IPG-BR" + "/nashconvs.npy"))
            if len(nashconvs) >= global_n_iter:
                print("IPG-BR already satisfies global_n_iter\n")
                n_iter = 0
            else:
                n_iter = global_n_iter - len(nashconvs)
                print(f"continue IPG-BR, n_iter={n_iter}\n")
        else:
            print("start IPG-BR\n")
            s0.build_random(gametree, random_seed)
            s1.build_random(gametree, random_seed)
            s2.build_random(gametree, random_seed)
            nashconvs = []
            n_iter = global_n_iter
        profile = StrategyProfile(gamerules, [s0,s1,s2], teams)

        for n in range(1, 1+n_iter):
            # best response
            br, ev = profile.best_response(br_players=[1])
            profile.strategies[1] = br.strategies[1]

            # gradient ascent
            br, ev = profile.best_response(ga_players=[0,2], ga_lr=lr)
            profile.strategies[0] = br.strategies[0]
            profile.strategies[2] = br.strategies[2]

            # evalute nashconv
            br, ev_br = profile.best_response(br_players=[0,1,2])
            nashconv = ev_br[0] + ev_br[1] + ev_br[2]
            nashconvs.append(nashconv)

            if n % 500 == 0:
                print(f"2v1Kuhn_r{rank}m{maxbets}seed{random_seed} IPG-BR checkpoint {n}: nashconv={nashconvs[-1]}")
                profile.strategies[0].save_to_file(save_path + "/IPG-BR" + "/p0.strat")
                profile.strategies[1].save_to_file(save_path + "/IPG-BR" + "/p1.strat")
                profile.strategies[2].save_to_file(save_path + "/IPG-BR" + "/p2.strat")
                np.save(save_path + "/IPG-BR" + "/nashconvs.npy", np.array(nashconvs))
       
        profile.strategies[0].save_to_file(save_path + "/IPG-BR" + "/p0.strat")
        profile.strategies[1].save_to_file(save_path + "/IPG-BR" + "/p1.strat")
        profile.strategies[2].save_to_file(save_path + "/IPG-BR" + "/p2.strat")
        np.save(save_path + "/IPG-BR" + "/nashconvs.npy", np.array(nashconvs))

    if "IBR-BR" in algo_list:
        s0 = Strategy(0)
        s1 = Strategy(1)
        s2 = Strategy(2)
        if os.path.exists(save_path + "/IBR-BR"):
            s0.load_from_file(save_path + "/IBR-BR" + "/p0.strat")
            s1.load_from_file(save_path + "/IBR-BR" + "/p1.strat")
            s2.load_from_file(save_path + "/IBR-BR" + "/p2.strat")
            nashconvs = list(np.load(save_path + "/IBR-BR" + "/nashconvs.npy"))
            if len(nashconvs) >= global_n_iter:
                print("IBR-BR already satisfies global_n_iter\n")
                n_iter = 0
            else:
                n_iter = global_n_iter - len(nashconvs)
                print(f"continue IBR-BR, n_iter={n_iter}\n")
        else:
            print("start IBR-BR\n")
            s0.build_random(gametree, random_seed)
            s1.build_random(gametree, random_seed)
            s2.build_random(gametree, random_seed)
            nashconvs = []
            n_iter = global_n_iter
        profile = StrategyProfile(gamerules, [s0,s1,s2], teams)

        for n in range(1, 1+n_iter):
            # best response
            br, ev = profile.best_response(br_players=[1])
            profile.strategies[1] = br.strategies[1]

            # best response
            br, _ = profile.best_response(br_players=[0,2])
            profile.strategies[0] = br.strategies[0]
            profile.strategies[2] = br.strategies[2]

            # evaluate NashConv
            br, ev_br = profile.best_response(br_players=[0,1,2])
            nashconv = ev_br[0] + ev_br[1] + ev_br[2]
            nashconvs.append(nashconv)

            if n % 500 == 0:
                print(f"2v1Kuhn_r{rank}m{maxbets}seed{random_seed} IBR-BR checkpoint {n}: nashconv={nashconv}")
                profile.strategies[0].save_to_file(save_path + "/IBR-BR" + "/p0.strat")
                profile.strategies[1].save_to_file(save_path + "/IBR-BR" + "/p1.strat")
                profile.strategies[2].save_to_file(save_path + "/IBR-BR" + "/p2.strat")
                np.save(save_path + "/IBR-BR" + "/nashconvs.npy", np.array(nashconvs))

        profile.strategies[0].save_to_file(save_path + "/IBR-BR" + "/p0.strat")
        profile.strategies[1].save_to_file(save_path + "/IBR-BR" + "/p1.strat")
        profile.strategies[2].save_to_file(save_path + "/IBR-BR" + "/p2.strat")
        np.save(save_path + "/IBR-BR" + "/nashconvs.npy", np.array(nashconvs))

    if "Cycle BR" in algo_list:
        s0 = Strategy(0)
        s1 = Strategy(1)
        s2 = Strategy(2)
        if os.path.exists(save_path + "/Cycle BR"):
            s0.load_from_file(save_path + "/Cycle BR" + "/p0.strat")
            s1.load_from_file(save_path + "/Cycle BR" + "/p1.strat")
            s2.load_from_file(save_path + "/Cycle BR" + "/p2.strat")
            nashconvs = list(np.load(save_path + "/Cycle BR" + "/nashconvs.npy"))
            if len(nashconvs) >= global_n_iter:
                print("Cycle BR already satisfies global_n_iter\n")
                n_iter = 0
            else:
                n_iter = global_n_iter - len(nashconvs)
                print(f"continue Cycle BR, n_iter={n_iter}\n")
        else:
            print("start Cycle BR\n")
            s0.build_random(gametree, random_seed)
            s1.build_random(gametree, random_seed)
            s2.build_random(gametree, random_seed)
            nashconvs = []
            n_iter = global_n_iter
        profile = StrategyProfile(gamerules, [s0,s1,s2], teams)

        for n in range(1, 1+n_iter):
            # best response
            br, ev_br = profile.best_response(br_players=[0,1,2])
            profile.strategies[0] = br.strategies[0]
            profile.strategies[1] = br.strategies[1]
            profile.strategies[2] = br.strategies[2]
            nashconv = ev_br[0] + ev_br[1] + ev_br[2]
            nashconvs.append(nashconv)

            if n % 500 == 0:
                print(f"2v1Kuhn_r{rank}m{maxbets}seed{random_seed} Cycle BR checkpoint {n}: nashconv={nashconv}")
                profile.strategies[0].save_to_file(save_path + "/Cycle BR" + "/p0.strat")
                profile.strategies[1].save_to_file(save_path + "/Cycle BR" + "/p1.strat")
                profile.strategies[2].save_to_file(save_path + "/Cycle BR" + "/p2.strat")
                np.save(save_path + "/Cycle BR" + "/nashconvs.npy", np.array(nashconvs))

        profile.strategies[0].save_to_file(save_path + "/Cycle BR" + "/p0.strat")
        profile.strategies[1].save_to_file(save_path + "/Cycle BR" + "/p1.strat")
        profile.strategies[2].save_to_file(save_path + "/Cycle BR" + "/p2.strat")
        np.save(save_path + "/Cycle BR" + "/nashconvs.npy", np.array(nashconvs))
    

    with open(save_path + "/info.txt", "a") as f:
        f.write("\nexperiment: 2v1Kuhn_r{}m{} (#terminal={}, #infoset={}, depth={})\n".format(rank, maxbets, gametree.num_terminal_nodes, len(gametree.information_sets), depth))
        f.write("algorithms: {}\n".format(algo_list))
        f.write("hyperparameters: n_iter={}, lr={}, random_seed={}\n".format(global_n_iter, lr, random_seed))

    print("finish 2v1Kuhn_r{}m{}: #terminal={}, #infoset={}, depth={}".format(rank, maxbets, gametree.num_terminal_nodes, len(gametree.information_sets), depth))
    print("algorithms: {}".format(algo_list))
    print("hyperparameters: n_iter={}, lr={}, random_seed={}".format(global_n_iter, lr, random_seed))
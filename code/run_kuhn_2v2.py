from pokertrees import *
from pokergames import *
from pokerstrategy import *
from card import Card
from pokerstrategy import Strategy, StrategyProfile
from pokertrees import GameTree
import argparse

import matplotlib.pyplot as plt
import numpy as np

def parse_list(input_str):
    return input_str.split(',')

def sum_dict(dict1, dict2, dict3, weights):
    res = {}
    for key in dict1.keys():
            res[key] = [dict1[key][i] * weights[0] + dict2[key][i] * weights[1] + dict3[key][i] * weights[2] if dict1[key][i] != None else None for i in range(3)]
    return res

def policy_gradient_descent(strategy, grad, lr):
    for infoset in grad.keys():
        illegal = None
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
    4-player Kuhn Poker ([0,2] vs [1,3])
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=5, help="rank of the deck")
    parser.add_argument("--maxbets", type=int, default=1, help="max number of bets")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed of initial strategy")
    parser.add_argument("--n_iter", type=int, default=500, help="number of iterations")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--eval_interval", type=int, default=10, help="evaluation interval")
    parser.add_argument("--algo", type=parse_list, default=["ND", "IPG-MinBR", "IPG-MinBR Dual", "IPG-IBR", "IPG-IBR Dual", "IBR-MinBR", "IBR-IBR", "Cycle BR"], help="algorithms to run")
    parser.add_argument("--save_path_prefix", type=str, default="../results/2v2Kuhn", help="save path prefix")
    args = parser.parse_args()
    
    players = 4
    rank = args.rank
    maxbets = args.maxbets
    random_seed = args.random_seed

    global_n_iter = args.n_iter
    lr = args.lr

    assert rank > 4 and rank < 14
    deck = [Card(14-r,1) for r in range(rank)]
    betsize = 1 # 不改
    rounds = [RoundInfo(holecards=1,boardcards=0,betsize=betsize,maxbets=[maxbets,maxbets,maxbets,maxbets])] # betsize是每次raise的大小,maxbets表示每个玩家最多允许加注的数额
    ante = 1 # 不改
    blinds = None # kuhn扑克没有大盲小盲
    teams=[[0,2],[1,3]]
    gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format)
    gametree = GameTree(gamerules, teams)
    gametree.build()
    
    """
    available algorithms:
    ND
    IPG-IBR
    IPG-IBR Dual
    IPG-MinBR
    IPG-MinBR Dual
    IBR-MinBR
    IBR-IBR
    Cycle BR
    """
    algo_list = args.algo
    eval_interval = args.eval_interval
    save_path = args.save_path_prefix + "/r{}m{}lr{}/seed{}".format(rank, maxbets, lr, random_seed)

    depth = 0
    for s in gametree.information_sets:
        depth = max(depth, len(s))
    depth -= 3

    print("running 2v2Kuhn_r{}m{}: #terminal={}, #infoset={}, depth={}".format(rank, maxbets, gametree.num_terminal_nodes, len(gametree.information_sets), depth))
    print("algorithms: {}".format(algo_list))
    print("hyperparameters: n_iter={}, eval_interval={}, lr={}, random_seed={}\n".format(global_n_iter, eval_interval, lr, random_seed))

    for algo in algo_list:
        # initialize policy from scratch or load from checkpoint
        s0 = Strategy(0)
        s1 = Strategy(1)
        s2 = Strategy(2)
        s3 = Strategy(3)
        if os.path.exists(save_path + "/" + algo):
            s0.load_from_file(save_path + "/" + algo + "/p0.strat")
            s1.load_from_file(save_path + "/" + algo + "/p1.strat")
            s2.load_from_file(save_path + "/" + algo + "/p2.strat")
            s3.load_from_file(save_path + "/" + algo + "/p3.strat")
            nashconvs = list(np.load(save_path + "/" + algo + "/nashconvs.npy"))
            if len(nashconvs) * eval_interval >= global_n_iter:
                print(f"{algo} already satisfies global_n_iter\n")
                n_iter = 0
            else:
                n_iter = global_n_iter - len(nashconvs) * eval_interval
                print(f"continue {algo}, n_iter={n_iter}\n")
        else:
            print(f"start {algo}\n")
            s0.build_random(gametree, random_seed)
            s1.build_random(gametree, random_seed)
            s2.build_random(gametree, random_seed)
            s3.build_random(gametree, random_seed)
            nashconvs = []
            n_iter = global_n_iter
        profile = StrategyProfile(gamerules, [s0,s1,s2,s3], teams)

        if algo == "IPG-MinBR Dual" or algo == "IPG-IBR Dual":
            profile_A = StrategyProfile(gamerules, [profile.strategies[0], profile.strategies[1], profile.strategies[2], profile.strategies[3]], teams)
            profile_B = StrategyProfile(gamerules, [profile.strategies[0], profile.strategies[1], profile.strategies[2], profile.strategies[3]], teams)
        # run the algorithm
        for n in range(1, n_iter+1):
            if algo == "ND":
                # best response
                br, ev_br = profile.best_response(br_players=[0,1,2,3])
                profile_br3 = StrategyProfile(gamerules, [profile.strategies[0], profile.strategies[1], profile.strategies[2], br.strategies[3]], teams)
                profile_br2 = StrategyProfile(gamerules, [profile.strategies[0], profile.strategies[1], br.strategies[2], profile.strategies[3]], teams)
                profile_br1 = StrategyProfile(gamerules, [profile.strategies[0], br.strategies[1], profile.strategies[2], profile.strategies[3]], teams)
                profile_br0 = StrategyProfile(gamerules, [br.strategies[0], profile.strategies[1], profile.strategies[2], profile.strategies[3]], teams)
                nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                nashconvs.append(nashconv)

                worst_profile = StrategyProfile(gamerules, [profile.strategies[0], br.strategies[1], profile.strategies[2], br.strategies[3]], teams)
                worst_ev = worst_profile.expected_value()
                # update player 0
                grad1, _ = profile_br1.best_response(ga_players=[0])
                grad2, _ = profile_br2.best_response(ga_players=[0])
                grad3, _ = profile_br3.best_response(ga_players=[0])
                total_grad = sum_dict(grad1.strategies[0].policy, grad2.strategies[0].policy, grad3.strategies[0].policy, weights=[-1,1,-1])
                policy_gradient_descent(profile.strategies[0], total_grad, lr)
                # update player 1
                grad1, _ = profile_br0.best_response(ga_players=[1])
                grad2, _ = profile_br2.best_response(ga_players=[1])
                grad3, _ = profile_br3.best_response(ga_players=[1])
                total_grad = sum_dict(grad1.strategies[1].policy, grad2.strategies[1].policy, grad3.strategies[1].policy, weights=[-1,-1,1])
                policy_gradient_descent(profile.strategies[1], total_grad, lr)
                # update player 2
                grad1, _ = profile_br0.best_response(ga_players=[2])
                grad2, _ = profile_br1.best_response(ga_players=[2])
                grad3, _ = profile_br3.best_response(ga_players=[2])
                total_grad = sum_dict(grad1.strategies[2].policy, grad2.strategies[2].policy, grad3.strategies[2].policy, weights=[1,-1,-1])
                policy_gradient_descent(profile.strategies[2], total_grad, lr)
                # update player 3
                grad1, _ = profile_br0.best_response(ga_players=[3])
                grad2, _ = profile_br1.best_response(ga_players=[3])
                grad3, _ = profile_br2.best_response(ga_players=[3])
                total_grad = sum_dict(grad1.strategies[3].policy, grad2.strategies[3].policy, grad3.strategies[3].policy, weights=[-1,1,-1])
                policy_gradient_descent(profile.strategies[3], total_grad, lr)

            elif algo == "IPG-MinBR":
                br, ev_br = profile.best_response(br_players=[1,3])
                # evaluate
                if n % eval_interval == 0:
                    eval_profile = StrategyProfile(gamerules, [profile.strategies[0], br.strategies[1], profile.strategies[2], br.strategies[3]], teams)
                    ev = eval_profile.expected_value()

                    br, ev_br = profile.best_response(br_players=[0,1,2,3])
                    nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                    nashconvs.append(nashconv)
                if ev_br[1] > ev_br[3]: # take the worst opponent br, update it, and do IPG against it
                    profile.strategies[1] = br.strategies[1]
                else:
                    profile.strategies[3] = br.strategies[3]
                br, _ = profile.best_response(ga_players=[0,2], ga_lr=lr)
                profile.strategies[0] = br.strategies[0]
                profile.strategies[2] = br.strategies[2]

            elif algo == "IPG-MinBR Dual":
                # eval
                if n % eval_interval == 0:
                    profile = StrategyProfile(gamerules, [profile_A.strategies[0], profile_B.strategies[1], profile_A.strategies[2], profile_B.strategies[3]], teams)
                    _, ev_br = profile.best_response(br_players=[0,1,2,3])
                    nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                    nashconvs.append(nashconv)
                # left team
                br, ev_br = profile_A.best_response(br_players=[1,3])
                if ev_br[1] > ev_br[3]:
                    profile_A.strategies[1] = br.strategies[1]
                else:
                    profile_A.strategies[3] = br.strategies[3]
                br, _ = profile_A.best_response(ga_players=[0,2], ga_lr=lr)
                profile_A.strategies[0] = br.strategies[0]
                profile_A.strategies[2] = br.strategies[2]
                # right team
                br, ev_br = profile_B.best_response(br_players=[0,2])
                if ev_br[0] > ev_br[2]: 
                    profile_B.strategies[0] = br.strategies[0]
                else:
                    profile_B.strategies[2] = br.strategies[2]
                br, _ = profile_B.best_response(ga_players=[1,3], ga_lr=lr)
                profile_B.strategies[1] = br.strategies[1]
                profile_B.strategies[3] = br.strategies[3]

            elif algo == "IPG-IBR":
                # opponent team: independent BR
                br, ev_br = profile.best_response(br_players=[1,3])
                profile.strategies[1] = br.strategies[1]
                profile.strategies[3] = br.strategies[3]
                # evaluate
                if n % eval_interval == 0:
                    ev = profile.expected_value()

                    br, ev_br = profile.best_response(br_players=[0,1,2,3])
                    nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                    nashconvs.append(nashconv)

                # target team: independent PG
                br, _ = profile.best_response(ga_players=[0,2], ga_lr=lr)
                profile.strategies[0] = br.strategies[0]
                profile.strategies[2] = br.strategies[2]

            elif algo == "IPG-IBR Dual":
                # eval
                if n % eval_interval == 0:
                    profile = StrategyProfile(gamerules, [profile_A.strategies[0], profile_B.strategies[1], profile_A.strategies[2], profile_B.strategies[3]], teams)
                    _, ev_br = profile.best_response(br_players=[0,1,2,3])
                    nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                    nashconvs.append(nashconv)
                # left team
                br, _ = profile_A.best_response(br_players=[1,3])
                profile_A.strategies[1] = br.strategies[1]
                profile_A.strategies[3] = br.strategies[3]
                br, _ = profile_A.best_response(ga_players=[0,2], ga_lr=lr)
                profile_A.strategies[0] = br.strategies[0]
                profile_A.strategies[2] = br.strategies[2]
                # right team
                br, _ = profile_B.best_response(br_players=[0,2])
                profile_B.strategies[0] = br.strategies[0]
                profile_B.strategies[2] = br.strategies[2]
                br, _ = profile_B.best_response(ga_players=[1,3], ga_lr=lr)
                profile_B.strategies[1] = br.strategies[1]
                profile_B.strategies[3] = br.strategies[3]

            elif algo == "IBR-IBR":
                # opponent team: independent BR
                br, _ = profile.best_response(br_players=[1,3])
                profile.strategies[1] = br.strategies[1]
                profile.strategies[3] = br.strategies[3]             
                # evaluate
                if n % eval_interval == 0:
                    ev = profile.expected_value()
                    br, ev_br = profile.best_response(br_players=[0,1,2,3])
                    nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                    nashconvs.append(nashconv)
                # target team: independent BR
                br, _ = profile.best_response(br_players=[0,2])
                profile.strategies[0] = br.strategies[0]
                profile.strategies[2] = br.strategies[2]
                
            elif algo == "IBR-MinBR":
                br, ev_br = profile.best_response(br_players=[1,3])
                # evaluate
                if n % eval_interval == 0:
                    eval_profile = StrategyProfile(gamerules, [profile.strategies[0], br.strategies[1], profile.strategies[2], br.strategies[3]], teams)
                    ev = eval_profile.expected_value()
                    br, ev_br = profile.best_response(br_players=[0,1,2,3])
                    nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                    nashconvs.append(nashconv)
                if ev_br[1] > ev_br[3]:
                    profile.strategies[1] = br.strategies[1]
                else:
                    profile.strategies[3] = br.strategies[3]
                br, _ = profile.best_response(br_players=[0,2])
                profile.strategies[0] = br.strategies[0]
                profile.strategies[2] = br.strategies[2]

            elif algo == "Cycle BR":
                br, ev_br = profile.best_response(br_players=[0,1,2,3])
                profile.strategies[0] = br.strategies[0]
                profile.strategies[2] = br.strategies[2]
                profile.strategies[1] = br.strategies[1]
                profile.strategies[3] = br.strategies[3]  
                nashconv = ev_br[0] + ev_br[1] + ev_br[2] + ev_br[3]
                nashconvs.append(nashconv)
                if n % eval_interval == 0:
                    eval_profile = StrategyProfile(gamerules, [profile.strategies[0], br.strategies[1], profile.strategies[2], br.strategies[3]], teams)
                    ev = eval_profile.expected_value()

            else:
                raise NotImplementedError(f"algorithm {algo} not implemented")

            # save checkpoint
            if n % 500 == 0:
                print(f"2v2Kuhn_r{rank}m{maxbets}seed{random_seed} {algo} checkpoint: n={n}, nashconv={nashconvs[-1]}")
                profile.strategies[0].save_to_file(save_path + "/" + algo + "/p0.strat")
                profile.strategies[1].save_to_file(save_path + "/" + algo + "/p1.strat")
                profile.strategies[2].save_to_file(save_path + "/" + algo + "/p2.strat")
                profile.strategies[3].save_to_file(save_path + "/" + algo + "/p3.strat")
                np.save(save_path + "/" + algo + "/nashconvs.npy", np.array(nashconvs))
       
        profile.strategies[0].save_to_file(save_path + "/" + algo + "/p0.strat")
        profile.strategies[1].save_to_file(save_path + "/" + algo + "/p1.strat")
        profile.strategies[2].save_to_file(save_path + "/" + algo + "/p2.strat")
        profile.strategies[3].save_to_file(save_path + "/" + algo + "/p3.strat")
        np.save(save_path + "/" + algo + "/nashconvs.npy", np.array(nashconvs))


    with open(save_path + "/info.txt", "a") as f:
        f.write("\nexperiment: 2v2Kuhn_r{}m{} (#terminal={}, #infoset={}, depth={})\n".format(rank, maxbets, gametree.num_terminal_nodes, len(gametree.information_sets), depth))
        f.write("algorithms: {}\n".format(algo_list))
        f.write("hyperparameters: n_iter={}, eval_interval={}, lr={}, random_seed={}\n".format(global_n_iter, eval_interval, lr, random_seed))

    print("finish 2v2Kuhn_r{}m{}: #terminal={}, #infoset={}, depth={}".format(rank, maxbets, gametree.num_terminal_nodes, len(gametree.information_sets), depth))
    print("algorithms: {}".format(algo_list))
    print("hyperparameters: n_iter={}, eval_interval={}, lr={}, random_seed={}\n".format(global_n_iter, eval_interval, lr, random_seed))

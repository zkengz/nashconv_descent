from pokertrees import *
import random
import os
import numpy as np

def L2proj_simplex(v):
    """
    project v to the probability simplex using L2 norm
    """
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(v) + 1) > (sv - 1))[0][-1]
    theta = (sv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def legal_L2proj_simplex(probs, legal_actions):
    res = [0,0,0]
    legal_probs = [probs[a] for a in range(3) if a in legal_actions]
    legal_probs = L2proj_simplex(legal_probs)
    j = 0
    for a in range(3):
        if a in legal_actions:
            res[a] = legal_probs[j]
            j += 1
    return res

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

class Strategy(object):
    def __init__(self, player, filename=None):
        self.player = player
        self.policy = {}
        if filename is not None:
            self.load_from_file(filename)

    def build_default(self, gametree):
        for key in gametree.information_sets:
            infoset = gametree.information_sets[key]
            test_node = infoset[0]
            if test_node.player == self.player:
                for node in infoset:
                    prob = 1.0 / float(len(node.children))
                    probs = [0,0,0]
                    for action in range(3):
                        if node.valid(action):
                            probs[action] = prob
                    if type(node.player_view) is tuple:
                        for pview in node.player_view:
                            self.policy[pview] = [x for x in probs]
                    else:
                        self.policy[node.player_view] = probs

    def build_random(self, gametree, seed=0):
        random.seed(seed)
        for key in gametree.information_sets:
            infoset = gametree.information_sets[key]
            test_node = infoset[0]
            if test_node.player == self.player:
                for node in infoset:
                    probs = [0 for _ in range(3)]
                    total = 0
                    for action in range(3):
                        if node.valid(action):
                            probs[action] = random.random()
                            total += probs[action]
                    probs = [x / total for x in probs]
                    if type(node.player_view) is tuple:
                        for pview in node.player_view:
                            self.policy[pview] = [x for x in probs]
                    else:
                        self.policy[node.player_view] = probs

    def probs(self, infoset):
        assert(infoset in self.policy)
        return self.policy[infoset]

    def sample_action(self, infoset):
        assert(infoset in self.policy)
        probs = self.policy[infoset]
        val = random.random()
        total = 0
        for i,p in enumerate(probs):
            total += p
            if p > 0 and val <= total:
                return i
        raise Exception('Invalid probability distribution. Infoset: {0} Probs: {1}'.format(infoset, probs))

    def load_from_file(self, filename):
        self.policy = {}
        f = open(filename, 'r')
        for line in f:
            line = line.strip()
            if line == "" or line.startswith('#'):
                continue
            tokens = line.split(' ')
            assert(len(tokens) == 4)
            key = tokens[0]
            probs = [float(x) for x in reversed(tokens[1:])]
            self.policy[key] = probs

    def save_to_file(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        f = open(filename, 'w')
        for key in sorted(self.policy.keys()):
            val = self.policy[key]
            f.write("{0} {1:.9f} {2:.9f} {3:.9f}\n".format(key, val[2], val[1], val[0]))
        f.flush()
        f.close()

class StrategyProfile(object):
    def __init__(self, rules, strategies, teams=None, gradient_only=False):
        assert(rules.players == len(strategies))
        self.rules = rules
        self.strategies = strategies
        self.gametree = None
        self.publictree = None
        self.teams = teams
        self.gradient_only = gradient_only # whether to update strategy in gradient ascent

    def expected_value(self):
        """
        Calculates the expected value of each strategy in the profile.
        Returns an array of scalars corresponding to the expected payoffs.
        """
        if self.gametree is None:
            self.gametree = PublicTree(self.rules, self.teams)
        if self.gametree.root is None:
            self.gametree.build()
        expected_values = self.ev_helper(self.gametree.root, [{(): 1} for _ in range(self.rules.players)])
        for ev in expected_values:
            assert(len(ev) == 1)
        return tuple(list(ev.values())[0] for ev in expected_values) # pull the EV from the dict returned

    def ev_helper(self, root, reachprobs):
        if type(root) is TerminalNode:
            return self.ev_terminal_node(root, reachprobs)
        if type(root) is HolecardChanceNode:
            return self.ev_holecard_node(root, reachprobs)
        if type(root) is BoardcardChanceNode:
            return self.ev_boardcard_node(root, reachprobs)
        return self.ev_action_node(root, reachprobs)

    def ev_terminal_node(self, root, reachprobs, counterfactual=True):
        payoffs = [None for _ in range(self.rules.players)]
        if counterfactual:
            # q_i,pi(s) = [Σ_z∈S\ eta_pi_-i(h) * u_i(z)] / |S|
            for player in range(self.rules.players):
                player_payoffs = {hc: 0 for hc in root.holecards[player]}
                counts = {hc: 0 for hc in root.holecards[player]}
                for hands,winnings in root.payoffs.items():
                    prob = 1.0
                    player_hc = None
                    for opp,hc in enumerate(hands):
                        if opp == player:
                            player_hc = hc
                        else:
                            prob *= reachprobs[opp][hc]
                    player_payoffs[player_hc] += prob * winnings[player]
                    counts[player_hc] += 1
                for hc,count in counts.items():
                    if count > 0:
                        player_payoffs[hc] /= float(count)
                payoffs[player] = player_payoffs
        else:
            raise NotImplementedError("Non-counterfactual EV not implemented.")
        return payoffs

    def ev_holecard_node(self, root, reachprobs):
        assert(len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        next_reachprobs = [{ hc: reachprobs[player][hc[0:prevlen]] / possible_deals for hc in root.children[0].holecards[player] } for player in range(self.rules.players)]
        subpayoffs = self.ev_helper(root.children[0], next_reachprobs)
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand,winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def ev_boardcard_node(self, root, reachprobs):
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for bc in root.children:
            next_reachprobs = [{ hc: reachprobs[player][hc] / possible_deals for hc in bc.holecards[player] } for player in range(self.rules.players)]
            subpayoffs = self.ev_helper(bc, next_reachprobs)
            for player,subpayoff in enumerate(subpayoffs):
                for hand,winnings in subpayoff.items():
                    payoffs[player][hand] += winnings
        return payoffs

    def ev_action_node(self, root, reachprobs):
        strategy = self.strategies[root.player]
        next_reachprobs = deepcopy(reachprobs)
        action_probs = { hc: strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history)) for hc in root.holecards[root.player] }
        action_payoffs = [None, None, None]
        if root.fold_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][FOLD] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[FOLD] = self.ev_helper(root.fold_action, next_reachprobs)
        if root.call_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][CALL] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[CALL] = self.ev_helper(root.call_action, next_reachprobs)
        if root.raise_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][RAISE] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[RAISE] = self.ev_helper(root.raise_action, next_reachprobs)
        payoffs = []
        for player in range(self.rules.players):
            player_payoffs = { hc: 0 for hc in root.holecards[player] }
            for action,subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                if root.player == player:
                    for hc,winnings in subpayoff[player].items():
                        player_payoffs[hc] += winnings * action_probs[hc][action] # v_i_pi(s) = Σ_a∈A\ q_i_pi(s,a) * pi_i(a|s) 
                else:
                    for hc,winnings in subpayoff[player].items():
                        player_payoffs[hc] += winnings
            payoffs.append(player_payoffs)
        return payoffs

    def best_response(self, br_players=[], ga_players=[], ga_lr=1):
        """
        Calculates the best response OR gradient ascent response for each player in the strategy profile.
        Returns a list of tuples of the best response strategy and its expected value for each player.

        if a player P is not in br_players nor ga_players, 
        then the corresponding returned br is empty and returned ev is the expected value of P's current strategy. 
        """
        # br_players = [x for x in list(range(self.rules.players)) if x not in ga_players]
        if self.publictree is None:
            self.publictree = PublicTree(self.rules, self.teams)
        if self.publictree.root is None:
            self.publictree.build()
        responses = [Strategy(player) for player in range(self.rules.players)]
        expected_values = self.br_helper(self.publictree.root, [{(): 1} for _ in range(self.rules.players)], responses, br_players, ga_players, ga_lr)
        for ev in expected_values:
            assert(len(ev) == 1)
        expected_values = tuple(list(ev.values())[0] for ev in expected_values) # pull the EV from the dict returned
        return (StrategyProfile(self.rules, responses, self.teams), expected_values)

    def br_helper(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        if type(root) is TerminalNode:
            return self.ev_terminal_node(root, reachprobs)
        if type(root) is HolecardChanceNode:
            return self.br_holecard_node(root, reachprobs, responses, br_players, ga_players, ga_lr)
        if type(root) is BoardcardChanceNode:
            return self.br_boardcard_node(root, reachprobs, responses, br_players, ga_players, ga_lr)
        return self.br_action_node(root, reachprobs, responses, br_players, ga_players, ga_lr)

    def br_holecard_node(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        assert(len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        next_reachprobs = [{ hc: reachprobs[player][hc[0:prevlen]] / possible_deals for hc in root.children[0].holecards[player] } for player in range(self.rules.players)]
        subpayoffs = self.br_helper(root.children[0], next_reachprobs, responses, br_players, ga_players, ga_lr)
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand,winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def br_boardcard_node(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for bc in root.children:
            next_reachprobs = [{ hc: reachprobs[player][hc] / possible_deals for hc in bc.holecards[player] } for player in range(self.rules.players)]
            subpayoffs = self.br_helper(bc, next_reachprobs, responses, br_players, ga_players, ga_lr)
            for player,subpayoff in enumerate(subpayoffs):
                for hand,winnings in subpayoff.items():
                    payoffs[player][hand] += winnings
        return payoffs

    def br_action_node(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        strategy = self.strategies[root.player]
        next_reachprobs = deepcopy(reachprobs)
        action_probs = { hc: strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history)) for hc in root.holecards[root.player] }
        action_payoffs = [None, None, None]
        if root.fold_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][FOLD] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[FOLD] = self.br_helper(root.fold_action, next_reachprobs, responses, br_players, ga_players, ga_lr)
        if root.call_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][CALL] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[CALL] = self.br_helper(root.call_action, next_reachprobs, responses, br_players, ga_players, ga_lr)
        if root.raise_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][RAISE] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[RAISE] = self.br_helper(root.raise_action, next_reachprobs, responses, br_players, ga_players, ga_lr)
        payoffs = []
        for player in range(self.rules.players):
            if player is root.player:
                if player in ga_players: # gradient ascent
                    if self.gradient_only: # compute gradient only, without updating strategy
                        payoffs.append(self.compute_gradient(root, responses, action_payoffs))
                    else: # compute gradient and update strategy
                        payoffs.append(self.ga_response_action(root, responses, action_payoffs, ga_lr))
                elif player in br_players: # best response
                    payoffs.append(self.br_response_action(root, responses, action_payoffs))
                else:
                    player_payoffs = { hc: 0 for hc in root.holecards[player] }
                    for action, subpayoff in enumerate(action_payoffs):
                        if subpayoff is None:
                            continue
                        for hc,winnings in subpayoff[player].items():
                            player_payoffs[hc] += winnings * action_probs[hc][action]
                    payoffs.append(player_payoffs)
            else:
                player_payoffs = { hc: 0 for hc in root.holecards[player] }
                for subpayoff in action_payoffs:
                    if subpayoff is None:
                        continue
                    for hc,winnings in subpayoff[player].items():
                        player_payoffs[hc] += winnings
                payoffs.append(player_payoffs)
        return payoffs

    def br_response_action(self, root, responses, action_payoffs):
        """
        compute best response strategy (parameter: repsonses) and return the corresponding payoffs
        """
        player_payoffs = { }
        max_strategy = responses[root.player]
        for hc in root.holecards[root.player]:
            max_action = None
            if action_payoffs[FOLD]:
                max_action = [FOLD]
                max_value = action_payoffs[FOLD][root.player][hc]
            if action_payoffs[CALL]:
                value = action_payoffs[CALL][root.player][hc]
                if max_action is None or value > max_value:
                    max_action = [CALL]
                    max_value = value
                elif max_value == value:
                    max_action.append(CALL)
            if action_payoffs[RAISE]:
                value = action_payoffs[RAISE][root.player][hc]
                if max_action is None or value > max_value:
                    max_action = [RAISE]
                    max_value = value
                elif max_value == value:
                    max_action.append(RAISE)
            probs = [0,0,0]
            for action in max_action:
                probs[action] = 1.0 / float(len(max_action))
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            max_strategy.policy[infoset] = probs
            player_payoffs[hc] = max_value
        return player_payoffs

    def ga_response_action(self, root, responses, action_payoffs, ga_lr):
        """
        compute gradient ascent strategy (parameter: repsonses) and return the corresponding payoffs
        """
        player_payoffs = { hc: 0 for hc in root.holecards[root.player] }
        max_strategy = responses[root.player]
        legal_actions = [action for action in range(3) if action_payoffs[action]]
        for hc in root.holecards[root.player]:
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            probs = self.strategies[root.player].policy[infoset]
            if action_payoffs[FOLD]:
                advantage = action_payoffs[FOLD][root.player][hc]
                probs[FOLD] = probs[FOLD] + advantage * ga_lr
            if action_payoffs[CALL]:
                advantage = action_payoffs[CALL][root.player][hc]
                probs[CALL] = probs[CALL] + advantage * ga_lr
            if action_payoffs[RAISE]:
                advantage = action_payoffs[RAISE][root.player][hc]
                probs[RAISE] = probs[RAISE] + advantage * ga_lr
            max_strategy.policy[infoset] = legal_L2proj_simplex(probs, legal_actions)
            for action,subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                player_payoffs[hc] += subpayoff[root.player][hc] * probs[action]
        return player_payoffs

    def compute_gradient(self, root, responses, action_payoffs):
        """
        compute strategy gradient (para: repsonses) and return the corresponding payoffs
        """
        player_payoffs = { hc: 0 for hc in root.holecards[root.player] }
        gradients = responses[root.player]
        for hc in root.holecards[root.player]:
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            probs = self.strategies[root.player].policy[infoset]
            gradient = [None, None, None]
            if action_payoffs[FOLD]:
                cf_value = action_payoffs[FOLD][root.player][hc]
                gradient[FOLD] = cf_value
            if action_payoffs[CALL]:
                cf_value = action_payoffs[CALL][root.player][hc]
                gradient[CALL] = cf_value
            if action_payoffs[RAISE]:
                cf_value = action_payoffs[RAISE][root.player][hc]
                gradient[RAISE] = cf_value
            gradients.policy[infoset] = gradient
            for action,subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                player_payoffs[hc] += subpayoff[root.player][hc] * probs[action]
        return player_payoffs

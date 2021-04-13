import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from information import mi2eff


class base_auction(nn.Module):
    def __init__(self, agents, c, r, datastream):
        super(base_auction, self).__init__()
        self.agents = agents
        self.datastream = datastream
        self.c = c
        self.r = r

    def run(self, data, logger):
        pass

class recsys_auction(base_auction): 
    def __init__(self, agents, datastream, activation='sigmoid'):
        super().__init__(agents, 0, 0, datastream)
        self.users = datastream.users
        self.uEmb = datastream.dataset.uEmb
        self.vEmb = datastream.dataset.vEmb
        self.sharpness = datastream.dataset.sharpness
        if activation == 'sigmoid':
            self.activation =  self.sigmoid_pCTR
        elif activation == 'id':
            self.activation = self._pCTR 
        elif activation == 'binary':
            self.activation = lambda x,y : torch.sign(self._pCTR(x,y))

    def interact(self, pCTR):
        interaction = 1.0 if torch.rand(1) < pCTR else 0.0
        self.logger['pCTR'].append(float(pCTR))
        self.logger['interaction'].append(int(interaction))
        return interaction, pCTR

    def _pCTR(self, rec, u):
        return torch.sum(self.vEmb[rec] * self.uEmb[u])

    def sigmoid_pCTR(self, rec, u):
        return torch.sigmoid(self.sharpness*self._pCTR(rec,u))
    
    def user_decision(self, data):
        u,_ = data
        u = u.squeeze()
        self.logger['x'].append(int(u))
        choice = self.users[u].choose()
        self.logger['choices'].append(int(choice))
        return u, choice
    
    def query_choice_agent(self, u, choice):
        rec = self.agents[choice].rec(u)
        self.logger['recs'].append(int(rec))
        self.logger['agents'][choice]['y_hat'].append(int(rec))
        return rec
    
    def query_all_agents(self, u):
        all_recs = []
        for a in self.agents:
            rec = a.rec(u)
            all_recs.append(rec)
            self.logger['agents'][a.id]['y_hat'].append(int(rec))
        return all_recs 
    
    def agent_logging(self):
        for i,agent in enumerate(self.agents):
            self.logger['agents'][agent.id]['reward'].append(int(agent.returns))
            self.logger['agents'][agent.id]['wins'].append(int(agent.total_recs))
            self.logger['agents'][agent.id]['uEmb'] = agent.U
            self.logger['agents'][agent.id]['vEmb'] = agent.V
            self.logger['agents'][agent.id]['records'] = agent.avg_r
            self.logger['agents'][agent.id]['counts'] = agent.n
            self.logger['agents'][agent.id]['random_tally'] = agent.random_tally
        #log the ground truth here as well
        if 'uEmb' not in self.logger.keys():
            self.logger['uEmb'] = self.uEmb
            self.logger['vEmb'] = self.vEmb
          
    def update(self, interaction, pCTR, choice, u, rec):
        self.agents[choice].update(u, rec, interaction)
        self.users[u].update(choice, interaction)
        self.agent_logging()

    def run(self, data, logger):
        u, choice = self.user_decision(data)
        rec = self.query_choice_agent(u, choice) 
        self.update(*self.interact(self.activation(rec,u)), choice, u, rec)        

class recsys_auction_with_baselines(recsys_auction):
    def __init__(self, agents, datastream, activation='sigmoid'):
        super().__init__(agents, datastream, activation=activation)
    
    def run(self, data, logger):
        u, choice = self.user_decision(data)
        rec = self.query_all_agents(u)[choice] 
        self.update(*self.interact(self.activation(rec,u)), choice, u, rec)    

class recsys_auction_debiased(recsys_auction_with_baselines):
    def __init__(self, agents, datastream, activation='sigmoid'):
        super().__init__(agents, datastream, activation=activation)

    def update(self, interaction, pCTR, choice, u, rec):
        rand_u = np.random.randint(len(self.users))
        rand_rec = self.query_choice_agent(rand_u, choice) 
        interaction, _ = self.interact(self.activation(rand_rec,u))
        self.agents[choice].update(rand_u, rand_rec, interaction)
        self.users[u].update(choice, interaction)
        self.agent_logging()

class classification_auction(base_auction):

    def __init__(self, agents, datastream, c=1, r=2):
        super().__init__(agents, c, r, datastream)

    def _process_data(self, data):
        x,y = data
        #x = x.reshape(-1,1)
        y = y.float()
        return x,y 

    def score_models(self, data):
        x,y = self._process_data(data)

        self.logger['x'].append(x)
        self.logger['y'].append(y)

        y_hats = []
        #compute predicted vals
        scores = []
        for i,a in enumerate(self.agents):
            a.get_reward(-self.c)
            y_hat = a.predict(x)
            y_hats.append(y_hat)
            scores.append(self.score(y,y_hat))
        self.logger['scores'].append(scores)
        return scores, y_hats, x, y

    def system_correctness(self, scores):
        correct_agents = []
        for i,score in enumerate(scores):
            if score == 1:
                correct_agents.append(self.agents[i])

        if correct_agents == []:
            self.logger['agg-correct'].append(False)
            correct_agents = self.agents
        else:
            self.logger['agg-correct'].append(True)   

        return correct_agents

    def user_decision(self, scores):
        correct_agents = self.system_correctness(scores)
        wid = torch.randint(len(correct_agents), (1,))[0]
        return correct_agents[wid]

    def update_winner(self, winner, x, y):
        self.logger['winner'].append(winner.id)
        winner.get_reward(self.r)
        winner.add_data(x,y)
        winner._update(x,y)
        winner.wins += 1

    def update_agents(self, y_hats):
        for i,agent in enumerate(self.agents):
            self.logger['agents'][agent.id]['reward'].append(agent.reward)
            self.logger['agents'][agent.id]['wins'].append(agent.wins)
            self.logger['agents'][agent.id]['y_hat'].append(y_hats[i])
            self.logger['agents'][agent.id][
                'dataset_counts'] = agent.dataset_counts
    
    def run(self, data, logger):
        scores, y_hats, x, y = self.score_models(data)
        winner = self.user_decision(scores)
        self.update_winner(winner, x, y)
        self.update_agents(y_hats)

    def score(self, y, y_hat):
        return  1 if y == y_hat else 0


class inefficient_classification_auction(classification_auction):
   
    def __init__(self, agents, datastream, c=1, r=2, alpha=1, mi=None):
        super().__init__(agents, datastream, c=c, r=r)
        if mi == None:
            self.alpha = alpha
        else:
            self.alpha = mi2eff(mi, len(agents), 1e-3)

    def system_correctness(self, scores, wid):
        corr_agent = super().system_correctness(scores)
        self.logger['agg-correct'][-1] = self.logger['agg-correct'][-1] and \
            wid in [a.id for a in corr_agent]

    def user_decision(self, scores):
       s_np = np.array(scores)
       softmin = np.exp(self.alpha*s_np)/sum(np.exp(self.alpha*s_np))
       wid = np.random.choice(range(len(s_np)), size=1, p=softmin)[0]
       self.system_correctness(scores, wid)
       return self.agents[wid]

class debiased_classification_auction(inefficient_classification_auction):
    def __init__(self, agents, datastream, c=1, r=2, alpha=1, mi=None):
        super().__init__(agents, datastream, c=c, r=r, alpha=alpha, mi=mi)

    def update_winner(self, winner, x, y):
        #need to debias the learning -- produce a random iid sample instead
        idx = np.random.randint(len(self.datastream.dataset))
        x,y = self.datastream.dataset[idx]
        #x = torch.tensor(x)
        x,y = self._process_data((
            torch.tensor(x).unsqueeze(0),torch.tensor([y])))
        super().update_winner(winner, x, y)

class regression_auction(inefficient_classification_auction):
    def __init__(self, agents, datastream, c=1, r=2, alpha=1):
        super().__init__(agents, datastream, c=c, r=r, alpha=alpha)
        self.alpha = -alpha  #alpha is negative for losses 

    def system_correctness(self, scores, wid):
        self.logger['agg-correct'].append(scores[wid])

    def score(self, y, y_hat): #for now just implement square error
        return torch.sum((y-y_hat)**2)


class bidding_classification_auction(classification_auction):

    def __init__(self, agents, c, r, datastream):
        super().__init__(agents, c, r, datastream)

    def run(self, data, logger):
        #run one step of mechanism
        x,y = data
        x = x.reshape(-1,1)
        y = y.float()

        logger['x'].append(x)
        logger['y'].append(y)

       
        y_hats = []
        bidders = set()
        #compute predicted vals
        scores = []
        for i,a in enumerate(self.agents):
            atarg = a.target(x)
            logger['agents'][i]['bid'].append(atarg)
            if atarg:
                a.get_reward(-self.c)
                bidders.add(a)
            y_hat = a.predict(x)
            y_hats.append(y_hat)
            s = self.score(y,y_hat)
            scores.append(self.score(y,y_hat))
        
        logger['scores'].append(scores)
        correct_agents = []
        for i,score in enumerate(scores):
            if score == 1 and self.agents[i] in bidders:
                correct_agents.append(self.agents[i])

        if correct_agents == []:
            logger['agg-correct'].append(False)
            correct_agents = list(bidders)
        else:
            logger['agg-correct'].append(True)
        wid = -1
        if len(correct_agents) > 0:
            wid = torch.randint(len(correct_agents), (1,))[0]
            winner = correct_agents[wid]
            logger['winner'].append(winner.id)
            winner.get_reward(self.r)
            winner.add_data(x,y)
            winner.wins += 1
        else:
            logger['winner'].append(-1)

        logger['bidders'].append(bidders)
        for i,agent in enumerate(self.agents):
            net_reward = -self.c
            net_reward += self.r if  wid == agent.id else 0
            agent.update(net_reward)
            logger['agents'][agent.id]['reward'].append(agent.reward)
            logger['agents'][agent.id]['wins'].append(agent.wins)
            logger['agents'][agent.id]['y_hat'].append(y_hats[i])
            logger['agents'][agent.id]['state'].append(agent.dist_bucket_est)

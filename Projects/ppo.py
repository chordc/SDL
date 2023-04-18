import torch
import numpy as np
import time


class PPO():
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347)).
    refer to github link: https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/ppo.py"""

    def __init__(self, acmodel, 
                 batch_reseter,
                 gamma_decay=0.7,
                 verbose=True,
                 device=None, 
                 discount=0.99,
                 stop_reward=1., 
                 lr=0.001, 
                 gae_lambda=0.95, 
                 entropy_coef=0.01,
                 value_loss_coef=0.5, 
                 max_grad_norm=0.5, 
                 adam_eps=1e-8,
                 clip_eps=0.2,
                 recurrence=4,
                 preprocess_obss=None, 
                 reshape_reward=None,
                 run_logger=None,
                 run_visualizer=None):
        """
        Parameters:
        ----------
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        self.acmodel = acmodel
        self.device = device
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gamma_decay = gamma_decay
        self.max_grad_norm = max_grad_norm
        self.adam_eps = adam_eps
        self.clip_eps = clip_eps
        self.preprocess_obss = preprocess_obss
        self.reshape_reward = reshape_reward
        self.batch_reseter = batch_reseter
        self.run_logger = run_logger
        self.run_visualizer = run_visualizer
        self.verbose = verbose
        self.stop_reward = stop_reward
        self.optimizer=torch.optim.Adam(self.acmodel.parameters(),lr=self.lr,eps=self.adam_eps)
        self.logits=[]
        self.actions=[]
        self.values=[]
        #self.log_probs=[]
        self.rewards=[]
        self.advantages=[]
        self.state=[]
        self.observations=[]
        self.recurrence=recurrence
        self.batch=self.batch_reseter()
        self.batch_size=self.batch.batch_size
        self.max_time_step=self.batch.max_time_step

        self.states= self.acmodel.get_zeros_initial_state(self.batch_size)
        # Optimizer reset
        #self.optimizer.zero_grad()

    

    def collect_experiences(self):
        #exps : DictList

        for i in range(self.max_time_step):
            #observations
            observations = torch.tensor(self.batch.get_obs().astype(np.float32), requires_grad=False,)
            
            #do one agent-environment interaction
            res,value, states= self.acmodel(input_tensor=observations,states=self.states)

            #self.states=memory

            #save memories
            self.state.append(self.states)
            self.states=states

            #save observations
            self.observations.append(observations)

            #raw pro distribution
            outlogit=res

            #prior
            prior_array=self.batch.prior().astype(np.float32)

            #0 protection
            eplison=0
            prior_array[prior_array==0]=eplison

            #to log
            prior=torch.tensor(prior_array,requires_grad=False)
            logprior=torch.log(prior)

            #sample
            logit=outlogit+logprior
            logit= torch.where(torch.isnan(logit), torch.full_like(logit, 0), logit)
            logit= torch.where(torch.isinf(logit), torch.full_like(logit, 0), logit)
            action=torch.multinomial(torch.exp(logit),num_samples=1)[:,0]

            #log_prob=dist.log_prob(action)

            #save actions
            self.actions.append(action)
            self.logits.append(logit)
            #self.log_probs.append(log_prob)

            #informing embedding of new action
            self.batch.programs.append(action.detach().cpu().numpy())

            #compute reward
            if i==self.max_time_step-1:
                self.logits=torch.stack(self.logits,dim=0)
                self.actions=torch.stack(self.actions,dim=0)

                self.actions_array=self.actions.detach().cpu().numpy()
                          
                self.R=self.batch.get_rewards()
                reward=self.R
                reward=torch.from_numpy(reward).float()
            else:
                reward=torch.zeros(action.shape[0],dtype=torch.float32)
            
            #update values
            self.rewards.append(reward)
            self.values.append(value)
            advantage=torch.zeros(action.shape[0],dtype=torch.float32)
            self.advantages.append(advantage)

        #add advantage

        for i in reversed(range(self.max_time_step)):
            next_value=self.values[i+1] if i<self.max_time_step-1 else 0
            next_advantage=self.advantages[i+1] if i<self.max_time_step-1 else 0
            delta=self.rewards[i]+self.discount*next_value-self.values[i]
            self.advantages[i]=delta+self.discount*self.gae_lambda*next_advantage
            
        #define exp
        return 0
    def update_parameters(self,n_keep):
        #only keep best batch(candidates)
        keep=self.R.argsort()[::-1][0:n_keep].copy()
        self.keep=keep
        notkept=self.R.argsort()[::-1][n_keep:].copy()
        self.notkept=notkept
        #self.keep=keep

        #Elite candidates
        self.actions_array_train=self.actions_array[:,keep]
        ideal_probs_array_train=np.eye(self.batch.n_choices)[self.actions_array_train]

        #Elite candidates rewards
        R_train=torch.tensor(self.R[keep],requires_grad=False)  
        R_lim=R_train.min()

        #Elite candidates as one hot in torch
        ideal_probs_train=torch.tensor(ideal_probs_array_train.astype(np.float32),requires_grad=False)

        #Elite candidates advantages
        advantages_train=torch.stack(self.advantages,dim=0)[:,keep]

        #Elite candidates values
        values_train=torch.stack(self.values,dim=0)[:,keep]

        #Elite candidates log_probs
        #log_probs_train=torch.stack(self.log_probs,dim=0)[:,keep]

        #Elite candidates memories
        #memories_train=torch.stack(self.memories,dim=0)[:,:,:,keep]
       
        #Elite candidates observations
        #observations_train=torch.stack(self.observations,dim=0)[:,keep]

        #Elite candidates logits
        logits_train=self.logits[:,keep]

        #compute new dist, value, memory
        new_logits_trains=[]
        new_values_trains=[]
        #new_log_probs=[]
        #entropys=[]

        #reset new batch
        self.batch=self.batch_reseter()

        for i in range(self.max_time_step):
            new_res,new_value, _ = self.acmodel(input_tensor=self.observations[i],states=self.state[i])
            #prior
            prior_array=self.batch.prior().astype(np.float32)

            #new_log_prob=new_dist.log_prob(self.actions[i])
            #entropy=new_dist.entropy()

            #0 protection
            eplison=0
            prior_array[prior_array==0]=eplison

            #to log
            prior=torch.tensor(prior_array,requires_grad=False)
            logprior=torch.log(prior)

            #sample
            logit=new_res+logprior
            logit= torch.where(torch.isnan(logit), torch.full_like(logit, 0), logit)
            logit= torch.where(torch.isinf(logit), torch.full_like(logit, 0), logit)
            new_logits_trains.append(logit)
            new_values_trains.append(new_value)
            #new_log_probs.append(new_log_prob)
            #entropys.append(entropy)
        
        new_logits_trains=torch.stack(new_logits_trains,dim=0)[:,keep]
        new_values_trains=torch.stack(new_values_trains,dim=0)[:,keep]
        #new_log_probs=torch.stack(new_log_probs,dim=0)[:,keep]
        #entropys=torch.stack(entropys,dim=0)[:,keep]
        #sum_entropys=entropys.sum(dim=0)
        #sum entropy
        #entropy=entropys.sum()
        #Compute Loss
        lengths=self.batch.programs.n_lengths[keep]
        baseline=R_lim

        loss_val=self.loss_func(logits_train=logits_train,
                                new_logits_trains=new_logits_trains,
                                new_values_trains=new_values_trains,
                                #new_log_probs=new_log_probs,
                                values_train=values_train,
                                #log_probs_train=log_probs_train,
                                advantages_train=advantages_train,
                                ideal_probs_train=ideal_probs_train,
                                lengths=lengths,
                                #entropy=sum_entropys,
                                baseline=baseline,)
        return loss_val

    
    def safe_cross_entropy(self,p, logq, dim=-1):
        safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)#
        return -torch.sum(p * safe_logq, dim=dim)

    def loss_func(self,
                  logits_train,
                  new_logits_trains,
                  new_values_trains,
                  #new_log_probs,
                  values_train,
                  #log_probs_train,
                  advantages_train,
                  ideal_probs_train,
                  lengths,
                  #entropy,
                  baseline,
                  ):
        # Getting shape
        (max_time_step, n_train, n_choices,) = ideal_probs_train.shape

        #Length mask
        mask_length_np = np.tile(np.arange(0, max_time_step), (n_train, 1)  # (n_train, max_time_step,)
                             ).astype(int) < np.tile(lengths, (max_time_step, 1)).transpose()
        mask_length_np = mask_length_np.transpose().astype(float)  # (max_time_step, n_train,)
        mask_length = torch.tensor(mask_length_np, requires_grad=False)  # (max_time_step, n_train,)

        #Entropy mask
        entropy_gamma_decay = np.array([self.gamma_decay ** t for t in range(self.max_time_step)])  # (max_time_step,)
        entropy_decay_mask_np = np.tile(entropy_gamma_decay,
                                    (n_train, 1)).transpose() * mask_length_np  # (max_time_step, n_train,)
        entropy_decay_mask = torch.tensor(entropy_decay_mask_np, requires_grad=False)  # (max_time_step, n_train,)

        #Normaizing over action dim probs and logprobs
        probs=torch.nn.functional.softmax(logits_train,dim=2)
       
        log_probs=torch.log_softmax(logits_train,dim=2)
        next_log_probs=torch.log_softmax(new_logits_trains,dim=2)
        neglogp_per_step=self.safe_cross_entropy(ideal_probs_train,log_probs,dim=2) #(max_time_step, n_train,)
        next_neglogp_per_step=self.safe_cross_entropy(ideal_probs_train,next_log_probs,dim=2) #(max_time_step, n_train,)

        neglogp=torch.sum(neglogp_per_step*mask_length,dim=0)
        next_neglogp=torch.sum(next_neglogp_per_step*mask_length,dim=0)

        #Sum over action dim
        entropy_per_step=self.safe_cross_entropy(probs,log_probs,dim=2) #(max_time_step, n_train,)
        entropy=entropy_per_step.sum(dim=0).mean()
        #new_entropy_per_step=self.safe_cross_entropy(probs,next_log_probs,dim=2)

        # Sum over sequence dim, mean over batch dim

        #ratio=torch.exp(next_log_probs-log_probs)
        ratio=torch.exp(-next_neglogp+neglogp)
        self.ratio=ratio
        self.advantages_train=advantages_train
        surr1=ratio*advantages_train
        surr2=torch.clamp(ratio,1.0-self.clip_eps,1.0+self.clip_eps)*advantages_train
        policy_loss=-torch.min(surr1,surr2)
        #sum over sequence dim, mean over batch dim
        policy_loss=torch.sum(policy_loss*mask_length,dim=0).mean()

        value_clipped=values_train+torch.clamp(new_values_trains-values_train,-self.clip_eps,self.clip_eps)
        surr1=(new_values_trains-values_train-advantages_train+baseline).pow(2)
        surr2=(value_clipped-values_train-advantages_train+baseline).pow(2)
        #surr1=(new_values_trains-values_train-advantages_train).pow(2)
        #surr2=(value_clipped-values_train-advantages_train).pow(2)
        value_loss=torch.max(surr1,surr2)
        #sum over sequence dim, mean over batch dim
        value_loss=torch.sum(value_loss*mask_length,dim=0).mean()

        #loss=policy_loss+value_loss*self.value_loss_coef-entropy*self.entropy_coef
        loss=policy_loss+value_loss*self.value_loss_coef-entropy*self.entropy_coef
        mean_loss=loss.mean()
        return mean_loss


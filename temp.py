def all_Actr(popsan,o,batch_size):

    pi, logp_pi = popsan(o, batch_size)
    q_pi=[]
    for critic_idx in range(ac_kwargs["num_critic"]):
        q_pi.append( torch.mean( eval("ac.q%d(o, pi)"%(critic_idx+1)) ,-1) )
        #print(q_pi[critic_idx].shape)
    # q_pi = torch.min(q1_pi_targ, q2_pi_targ)
    q_pi = torch.min(torch.stack(q_pi,axis=-1),axis=-1).values
    #print(q_pi.shape)
    # q1_pi = torch.mean(ac.q1(o, pi),-1)
    # q2_pi = torch.mean(ac.q2(o, pi),-1)
    # q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
    #logp_pi=torch.stack([logp_pi,logp_pi,logp_pi],dim=1)
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.to('cpu').detach().numpy())
    return loss_pi, pi_info

    def all_Actr(popsan,o,batch_size):
    
        pi, logp_pi = popsan(o, batch_size)
        q_pi=[]
        for critic_idx in range(ac_kwargs["num_critic"]):
            q_pi.append( torch.mean( eval("ac.q%d(o, pi)"%(critic_idx+1)) ,-1) )
            #print(q_pi[critic_idx].shape)
        # q_pi = torch.min(q1_pi_targ, q2_pi_targ)
        q_pi = torch.min(torch.stack(q_pi,axis=-1),axis=-1).values
        #print(q_pi.shape)
        # q1_pi = torch.mean(ac.q1(o, pi),-1)
        # q2_pi = torch.mean(ac.q2(o, pi),-1)
        # q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        #logp_pi=torch.stack([logp_pi,logp_pi,logp_pi],dim=1)
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.to('cpu').detach().numpy())
        return loss_pi, pi_info

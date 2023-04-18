import torch
import numpy as np
import time
import ppo
# Internal imports
def learner_ppo ( model,
             n_epochs,
             batch_reseter,
             risk_factor,
             verbose = True,
             stop_reward = 1.,
             stop_after_n_epochs = 50,
             run_logger     = None,
             run_visualiser = None,
            ):
    """
    Trains model to generate symbolic programs satisfying a reward by reinforcing on best candidates at each epoch.
    Parameters
    ----------
    model : torch.nn.Module
        Differentiable RNN cell.
    optimizer : torch.optim
        Optimizer to use.
    n_epochs : int
        Number of epochs.
    batch_reseter : callable
        Function returning a new empty physym.batch.Batch.
    risk_factor : float
        Fraction between 0 and 1 of elite programs to reinforce on.
    gamma_decay : float
        Weight of power law to use along program length: gamma_decay**t where t is the step in the sequence in the loss
        function (gamma_decay < 1 gives more important to first tokens and gamma_decay > 1 gives more weight to last
        tokens).
    entropy_weight : float
        Weight to give to entropy part of the loss.
    verbose : int, optional
        If verbose = False or 0, print nothing, if True or 1, prints learning time, if > 1 print epochs progression.
    stop_reward : float, optional
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    stop_after_n_epochs : int, optional
        Number of additional epochs to do after early stop condition is reached.
    run_logger : object or None, optional
        Custom run logger to use having a run_logger.log method taking as args (epoch, batch, model, rewards, keep,
        notkept, loss_val).
    run_visualiser : object or None, optional
        Custom run visualiser to use having a run_visualiser.visualise method taking as args (run_logger, batch).
    Returns
    -------
    hall_of_fame_R, hall_of_fame : list of float, list of physym.program.Program
        hall_of_fame : history of overall best programs found.
        hall_of_fame_R : Corresponding reward values.
        Use hall_of_fame[-1] to access best model found.
    """
    t000 = time.perf_counter()

    # Basic logs
    overall_max_R_history = []
    hall_of_fame          = []

    for epoch in range (n_epochs):

        if verbose>1: print("Epoch %i/%i"%(epoch, n_epochs))

        # -------------------------------------------------
        # --------------------- INIT  ---------------------
        # -------------------------------------------------

        # Reset new batch (embedding reset)
        ppo1=ppo.PPO(acmodel=model,batch_reseter=batch_reseter)
        # Number of elite candidates to keep
        n_keep = int(risk_factor*ppo1.batch_size)

        # -------------------------------------------------
        # -------------------- RNN RUN  -------------------
        # -------------------------------------------------

        # RNN run
       
        ppo1.collect_experiences()

        # -------------------------------------------------
        # ---------------- BEST CANDIDATES ----------------
        # -------------------------------------------------

        
        # ----------------- Train batch : black box part (NUMPY) -----------------
        # Loss
        loss_val = ppo1.update_parameters(n_keep)

        # -------------------------------------------------
        # ---------------- BACKPROPAGATION ----------------
        # -------------------------------------------------
        # No need to do backpropagation if model is lobotomized (ie. is just a random number generator).
        if model.is_lobotomized:
            pass
        else:
            ppo1.optimizer.zero_grad()
            loss_val  .backward()
            #grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in ppo1.acmodel.parameters()) ** 0.5
            #torch.nn.utils.clip_grad_norm_(ppo1.acmodel.parameters(), ppo1.max_grad_norm)
            ppo1.optimizer .step()

        # -------------------------------------------------
        # ----------------- LOGGING VALUES ----------------
        # -------------------------------------------------

        # Basic logging (necessary for early stopper)
        if epoch == 0:
            overall_max_R_history       = [ppo1.R.max()]
            hall_of_fame                = [ppo1.batch.programs.get_prog(ppo1.R.argmax())]
        if epoch> 0:
            if ppo1.R.max() > np.max(overall_max_R_history):
                overall_max_R_history.append(ppo1.R.max())
                hall_of_fame.append(ppo1.batch.programs.get_prog(ppo1.R.argmax()))
            else:
                overall_max_R_history.append(overall_max_R_history[-1])

        # Custom logging
        if run_logger is not None:
            run_logger.log(epoch    = epoch,
                           batch    = ppo1.batch,
                           model    = model,
                           rewards  = ppo1.R,
                           keep     = ppo1.keep,
                           notkept  = ppo1.notkept,
                           loss_val = loss_val)

        # -------------------------------------------------
        # ----------------- VISUALISATION -----------------
        # -------------------------------------------------

        # Custom visualisation
        if run_visualiser is not None:
            run_visualiser.visualise(run_logger = run_logger, batch =ppo1.batch)

        # -------------------------------------------------
        # -------------------- STOPPER --------------------
        # -------------------------------------------------
        early_stop_reward_eps = 2*np.finfo(np.float32).eps

        # If above stop_reward (+/- eps) stop after [stop_after_n_epochs] epochs.
        if (stop_reward - overall_max_R_history[-1]) <= early_stop_reward_eps:
            if stop_after_n_epochs == 0:
                try:
                    run_visualiser.save_visualisation()
                    run_visualiser.save_data()
                    run_visualiser.save_pareto_data()
                    run_visualiser.save_pareto_fig()
                except:
                    print("Unable to save last plots and data before stopping.")
                break
            stop_after_n_epochs -= 1

    t111 = time.perf_counter()
    if verbose:
        print("  -> Time = %f s"%(t111-t000))

    hall_of_fame_R = np.array(overall_max_R_history)
    return hall_of_fame_R, hall_of_fame

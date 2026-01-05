import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.validation_utils import val_G, val_G_image
from data.SimulationData import generate_multi_responses_multiY
from utils.basic_utils import setup_seed, sample_noise, calculate_gradient_penalty, discriminator_loss, generator_loss, l1_loss, l2_loss
from utils.plot_utils import plot_kde_2d, convert_generated_to_mnist_range,  visualize_mnist_digits, visualize_digits 

def train_WGR_fnn(D, G, D_solver, G_solver, loader_train, loader_val, noise_dim, Xdim, Ydim, batch_size,  J_size=50, 
                  noise_distribution='gaussian', noise_mu=None, noise_cov=None, noise_a=None, noise_b=None, noise_loc=None, noise_scale=None, 
                  multivariate=False, lambda_w=0.9, lambda_l=0.1, save_path='./M1/', model_type="M1", start_eva=1000,  eva_iter = 50,
                  num_epochs=10, num_samples=100, device='cuda', lr_decay=None, save_name = None,
                  lr_decay_step=5, lr_decay_gamma=0.1, save_last  = False, is_plot=False, plot_iter=500):
    """
    Train Wasserstein GAN Regression with Fully-Connected Neural Networks.
    
    Args:
        D: Discriminator model
        G: Generator model
        D_solver: Discriminator optimizer
        G_solver: Generator optimizer
        loader_train: Data loader for training set
        loader_val: Data loader for validation set
        noise_dim: Dimension of noise vector
        Xdim: Dimension of covariate X
        Ydim: Dimension of response Y
        batch_size: Batch size
        J_size: Generator projection size (default: 50)
        noise_distribution: Distribution for noise sampling (default: 'gaussian')
        noise_mu (torch.Tensor): Mean vector for multivariate Gaussian
        noise_cov (torch.Tensor): Covariance matrix for multivariate Gaussian
        noise_a (float): Lower bound for uniform distribution
        noise_b (float): Upper bound for uniform distribution
        noise_loc (torch.Tensor): Location parameter for Laplace distribution
        noise_scale (torch.Tensor): Scale parameter for Laplace distribution
        lambda_w: Weight for Wasserstein loss (default: 0.9)
        lambda_l: Weight for L2 regularization (default: 0.1)
        save_path: Path to save models (default: './M1/')
        start_eva: Iteration to start evaluation (default: 1000)
        eva_iter: to conduct the validation per iteration (default: 50)
        num_epochs: Number of training epochs (default: 10)
        num_samples: Number of noise samples generated for each data point in validation (default: 100)
        device: Device to train on (default: 'cuda')
        lr_decay: Learning rate decay strategy ('step', 'plateau', 'cosine', or None)
        lr_decay_step: Step size for StepLR or patience for ReduceLROnPlateau
        lr_decay_gamma: Multiplicative factor for learning rate decay
        save_last: Whether to save the last trained network (default: False)
        is_plot: Whether to conduct visualization (default: False)
        plot_iter: to conduct the visualization per iteration (default: 500)
    Returns:
        tuple: Best validation scores and final models
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Move models to device
    D = D.to(device)
    G = G.to(device)
    
    # Initialize counters and metrics
    iter_count = 0
    l1_acc, l2_acc = val_G(G=G, loader_data=loader_val, noise_dim=noise_dim, Xdim=Xdim, Ydim=Ydim, distribution=noise_distribution , 
                           mu=noise_mu, cov=noise_cov, a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale, num_samples=num_samples, 
                           device=device,  multivariate=multivariate )
                         
    # Save initial model state
    best_acc = l2_acc
    best_model_g = copy.deepcopy(G.state_dict())
    best_model_d = copy.deepcopy(D.state_dict())
    
    # Initialize learning rate schedulers if requested
    D_scheduler, G_scheduler = None, None
    if lr_decay == 'step':
        D_scheduler = torch.optim.lr_scheduler.StepLR(
            D_solver, step_size=lr_decay_step, gamma=lr_decay_gamma)
        G_scheduler = torch.optim.lr_scheduler.StepLR(
            G_solver, step_size=lr_decay_step, gamma=lr_decay_gamma)
    elif lr_decay == 'plateau':
        D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            D_solver, mode='min', factor=lr_decay_gamma, patience=lr_decay_step )
        G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            G_solver, mode='min', factor=lr_decay_gamma, patience=lr_decay_step )
    elif lr_decay == 'cosine':
        D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            D_solver, T_max=num_epochs, eta_min=0)
        G_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            G_solver, T_max=num_epochs, eta_min=0)
    
    for epoch in range(num_epochs):
        D.train()
        G.train()
        d_losses = []
        g_losses = []
        
        for batch_idx, (x, y) in enumerate(loader_train):
            if x.size(0) != batch_size:
                continue
                
            # Move data to the appropriate device
            x, y = x.to(device), y.to(device)
            
            # Sample noise
            eta = sample_noise(x.size(0), dim=noise_dim, distribution=noise_distribution , mu=noise_mu, 
                               cov=noise_cov, a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale).to(device)
            
            # Prepare inputs
            d_input = torch.cat([x.view(batch_size, Xdim), y.view(batch_size, Ydim)], dim=1)     
            g_input = torch.cat([x.view(batch_size, Xdim), eta], dim=1)
            
            # ==================== Train Discriminator ====================
            D_solver.zero_grad()
            logits_real = D(d_input)
            
            fake_y = G(g_input).detach()
            fake_images = torch.cat([x.view(batch_size, Xdim), fake_y.view(batch_size, Ydim)], dim=1)
                
            logits_fake = D(fake_images)
            
            penalty = calculate_gradient_penalty(D, d_input, fake_images, device)
            d_error = discriminator_loss(logits_real, logits_fake) + 10 * penalty
            d_error.backward()
            D_solver.step()
            d_losses.append(d_error.item())
             
            # ==================== Train Generator ====================
            G_solver.zero_grad()
            
            # First: Standard WGAN loss
            fake_y = G(g_input)
            fake_images = torch.cat([x.view(batch_size, Xdim), fake_y.view(batch_size, Ydim)], dim=1)
            logits_fake = D(fake_images)
            g_error_w = generator_loss(logits_fake)

            # Second: Generate multiple outputs and compute L2 loss against expected y
            if lambda_l>0:  #if lambda_l = 0, then it becomes the standard cWGAN
                # Initialize output tensor with dimensions that work for both cases
                g_output = torch.zeros([J_size, batch_size, max(1, Ydim)], device=device)
                for i in range(J_size):
                    eta = sample_noise(x.size(0), noise_dim, distribution=noise_distribution , 
                                       mu=noise_mu, cov=noise_cov, a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale).to(device)
                    g_input = torch.cat([x.view(batch_size, Xdim), eta], dim=1)
                    output = G(g_input)
                    g_output[i] = output.view(batch_size, -1)  # Reshape to [batch_size, Ydim] or [batch_size, 1]
                    
                # Reshape final result if Ydim=1 to match expected dimensions
                if Ydim == 1:
                    g_output = g_output.squeeze(-1)  # Remove the last dimension to get [J_size, batch_size]
                    # For univariate output, compute mean squared error directly
                    g_error_l = torch.mean((g_output.mean(dim=0) - y.view(batch_size))**2)
                else:
                    # For multivariate output, use MSE loss function
                    g_error_l = torch.mean(torch.sum((g_output.mean(dim=0) - y)**2, dim=1))
            else: 
                g_error_l = 0 

            
            
            #y_reshaped = y.view(batch_size, -1)  # Reshape to [batch_size, Ydim] or [batch_size, 1]
            #g_error_l = torch.mean((mean_pred - y_reshaped)**2)

            # Combined loss with wasserstein and L2 regularization
            g_error = lambda_w * g_error_w + lambda_l * g_error_l
          
            g_error.backward()
            G_solver.step()
            g_losses.append(g_error.item())
            
            # Increment iteration counter
            iter_count += 1

        # Per-epoch validation/checkpointing
        l1_acc, l2_acc = val_G(G=G, loader_data=loader_val, noise_dim=noise_dim, Xdim=Xdim,  Ydim=Ydim, distribution=noise_distribution , mu=noise_mu, cov=noise_cov, a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale, num_samples=num_samples, device=device,  multivariate=multivariate )
        print(f"Epoch {epoch}, Iter {iter_count}, "
              f"D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}, "
              f"L1: {l1_acc:.4f}, L2: {l2_acc:.4f}")
        if (save_last == False) and (l2_acc < best_acc):
            best_acc = l2_acc
            best_model_g = copy.deepcopy(G.state_dict())
            best_model_d = copy.deepcopy(D.state_dict())
            torch.save(G.state_dict(), f"{save_path}/G_"+model_type+"_d"+str(Xdim)+"_m"+str(noise_dim)+"_best.pth")
            torch.save(D.state_dict(), f"{save_path}/D_"+model_type+"_d"+str(Xdim)+"_m"+str(noise_dim)+"_best.pth")
            print(f"Saved best model with L2: {best_acc:.4f}")

                        
                         
        
        # Apply learning rate decay at the end of each epoch
        epoch_d_loss = np.mean(d_losses)
        epoch_g_loss = np.mean(g_losses)
        
        print(f"Epoch {epoch} - "
              f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}")

        valid_loss = epoch_d_loss if epoch_d_loss is not None else float('inf')

        if lr_decay == 'step' or lr_decay == 'cosine':
            if D_scheduler is not None:
                D_scheduler.step()
            if G_scheduler is not None:
                G_scheduler.step()
        elif lr_decay == 'plateau':
            if D_scheduler is not None:
                D_scheduler.step(valid_loss)
            if G_scheduler is not None:
                G_scheduler.step(l2_acc)  # Use validation L2 for generator
        
        # Print current learning rates
        if lr_decay:
            d_lr = D_solver.param_groups[0]['lr']
            g_lr = G_solver.param_groups[0]['lr']
            print(f"Epoch {epoch} - D LR: {d_lr:.6f}, G LR: {g_lr:.6f}")
    
    # For multivariate response model, save models at the end of the training
    if save_last==True :
        best_model_g = copy.deepcopy(G.state_dict())
        best_model_d = copy.deepcopy(D.state_dict())
        
        torch.save(G.state_dict(), f"{save_path}/G_"+model_type+"_d"+str(Xdim)+"_m"+str(noise_dim)+"_best.pth")
        torch.save(D.state_dict(), f"{save_path}/D_"+model_type+"_d"+str(Xdim)+"_m"+str(noise_dim)+"_best.pth")
        print(f"Saved best model with L2: {best_acc:.4f}")

    # Load the best model at the end of training
    G.load_state_dict(best_model_g)
    D.load_state_dict(best_model_d)
    
    return G, D

def train_WGR_image(D,G, D_solver,G_solver, Xdim, Ydim, noise_dim, loader_data , loader_val , batch_size,  
                    eg_x, eg_label, selected_indices, lambda_w=0.9, lambda_l=0.1, noise_distribution= 'gaussian', 
                    noise_mu=None, noise_cov=None, noise_a=None, noise_b=None, noise_loc=None, noise_scale=None, 
                    save_path='.', num_epochs=10, start_eva=1000,  eva_iter = 50, data_type ='mnist',
                    device='cpu', lr_decay=None, r_decay_step=5, lr_decay_gamma=0.1, is_image=False ):
    """
    Train Wasserstein GAN Regression with Fully-Connected Neural Networks.
    
    Args:
        D: Discriminator model
        G: Generator model
        D_solver: Discriminator optimizer
        G_solver: Generator optimizer
        noise_dim: Dimension of noise vector
        Xdim: Dimension of covariate X
        Ydim: Dimension of response Y
        batch_size: Batch size
        loader_data: Data loader for training set
        loader_val: Data loader for validation set
        eg_x: Sample used to show the reconstruction performance
        eg_label: label of the eg_x
        selected_indices: indices for eg_x to sort it from 0 to 1
        noise_distribution: Distribution for noise sampling (default: 'gaussian')
        lambda_w: Weight for Wasserstein loss  (default: 0.9)
        lambda_l: Weight for L2 regularization  (default: 0.1)
        save_path: Path to save models (default: './ ')
        start_eva: Iteration to start evaluation (default: 1000)
        eva_iter: to conduct the validation per iteration (default: 50)
        num_epochs: Number of training epochs (default: 10)
        num_samples: Number of noise samples generated for each data point in validation (default: 100)
        device: Device to train on (default: 'cpu')
        lr_decay: Learning rate decay strategy ('step', 'plateau', 'cosine', or None)
        lr_decay_step: Step size for StepLR or patience for ReduceLROnPlateau
        lr_decay_gamma: Multiplicative factor for learning rate decay
    Returns:
        tuple: Best validation scores and final models
    """
    # Initialize learning rate schedulers if requested
    D_scheduler, G_scheduler = None, None
    if lr_decay == 'step':
        D_scheduler = torch.optim.lr_scheduler.StepLR(
            D_solver, step_size=lr_decay_step, gamma=lr_decay_gamma)
        G_scheduler = torch.optim.lr_scheduler.StepLR(
            G_solver, step_size=lr_decay_step, gamma=lr_decay_gamma)
    elif lr_decay == 'plateau':
        D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            D_solver, mode='min', factor=lr_decay_gamma, patience=lr_decay_step )
        G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            G_solver, mode='min', factor=lr_decay_gamma, patience=lr_decay_step )
    elif lr_decay == 'cosine':
        D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            D_solver, T_max=num_epochs, eta_min=0)
        G_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            G_solver, T_max=num_epochs, eta_min=0)
        
    iter_count = 0 
    best_acc = 5
    
    best_model_g = copy.deepcopy(G.state_dict())
    best_model_d = copy.deepcopy(D.state_dict())
    
    for epoch in range(num_epochs):
        for batch_idx, (x,y, label) in enumerate(loader_data):
            if x.size(0) != batch_size:
                continue
    
            eta = sample_noise(x.size(0), noise_dim, distribution=noise_distribution, mu=noise_mu, cov=noise_cov, a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale)
            x_data = x.view(x.size(0),784)
            g_input = torch.cat([x_data,eta],dim=1)
            
            #train D
            D_solver.zero_grad()
            real_images = x.clone()
            real_images[:,:,7:19,7:19] = y
            logits_real = D(real_images)
            
            fake_y = G(g_input).view(x.size(0),1,12,12).detach()
            fake_images = x.clone()
            fake_images[:,:,7:19,7:19] = fake_y
            logits_fake = D(fake_images)
            
            penalty = calculate_gradient_penalty(D,real_images,fake_images,device, is_image=True)
            d_error = discriminator_loss(logits_real, logits_fake) + 10 * penalty
            d_error.backward() 
            D_solver.step()
            
            # train G
            G_solver.zero_grad()
            fake_y = G(g_input).view(x.size(0),1,12,12)
            fake_images[:,:,7:19,7:19] = fake_y
            
            gen_logits_fake = D(fake_images)
            g_error = lambda_w * generator_loss(gen_logits_fake) + lambda_l * l2_loss(fake_y,y)
            g_error.backward()
            G_solver.step()
            
            if (iter_count % eva_iter == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))

                if (iter_count >= start_eva):
                    l1_G_Acc, l2_G_Acc= val_G_image(G, loader_data=loader_val, noise_dim=noise_dim, Xdim=Xdim, Ydim=Ydim, 
                                                    distribution=noise_distribution , mu=noise_mu, cov=noise_cov, 
                                                    a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale, multivariate=True)
                    if l2_G_Acc < best_acc:
                        print('################## save G model #################')
                        best_acc = l2_G_Acc.copy()
                        best_model_g = copy.deepcopy(G.state_dict())
                        best_model_d = copy.deepcopy(D.state_dict())

                        # Save models
                        torch.save(G.state_dict(), f"{save_path}/G_"+data_type+"_d"+str(Xdim)+"_m"+str(noise_dim)+"_"+str(save_name)+"_best.pth")
                        torch.save(D.state_dict(), f"{save_path}/D_"+data_type+"_d"+str(Xdim)+"_m"+str(noise_dim)+"_"+str(save_name)+"_best.pth")
                        print(f"Saved best model with L2: {best_acc:.4f}")

                        # plot the reconstruction image on the examples
                        eg_eta =  sample_noise(batch_size, dim=noise_dim, distribution=noise_distribution , mu=noise_mu, 
                                               cov=noise_cov, a=noise_a, b=noise_b, loc=noise_loc, scale=noise_scale ).to(device)
                        g_exam_input = torch.cat([eg_x.view(batch_size, Xdim), eg_eta], dim=1)
                        recon_y = G(g_exam_input).view(batch_size,1,12,12)
                        recover_y = convert_generated_to_mnist_range(recon_y)
                        
                        recon_x = eg_x.clone()
                        recon_x[selected_indices,:,7:19,7:19] = recover_y[selected_indices,:,:,:].detach()
                        visualize_digits( images=recon_x[selected_indices] , labels = eg_label[selected_indices], figsize=(3, 13), title='(X,hat(Y)')
            iter_count += 1
          
        if lr_decay == 'step' or lr_decay == 'cosine':
            if D_scheduler is not None:
                D_scheduler.step()
            if G_scheduler is not None:
                G_scheduler.step()
        elif lr_decay == 'plateau':
            if D_scheduler is not None:
                D_scheduler.step(valid_loss)
            if G_scheduler is not None:
                G_scheduler.step(l2_acc)  # Use validation L2 for generator
        
        # Print current learning rates
        if lr_decay:
            d_lr = D_solver.param_groups[0]['lr']
            g_lr = G_solver.param_groups[0]['lr']
            print(f"Epoch {epoch} - D LR: {d_lr:.6f}, G LR: {g_lr:.6f}")      


    # Load the best model at the end of training
    G.load_state_dict(best_model_g)
    D.load_state_dict(best_model_d)
    
    return G, D


def train_dnls(net, solver, loader_data, loader_val, reps, num_epochs=10, Best_acc = 50):
    iter_count = 0 
    l1_Acc, l2_Acc = val_dnls(net=net, loader_val=loader_val)
    
    
    for epoch in range(num_epochs):
        for batch_idx, (x,y) in enumerate(loader_data):
            if x.size(0) != args.train_batch:
                continue
    
            solver.zero_grad()
            fake_y = net(x).view(x.size(0))
            
            dnls_error = l2_loss(fake_y, y)
            dnls_error.backward()
            solver.step()
            
            if(iter_count > 500):    
                if(iter_count % 100 == 0):
                    l1_Acc, l2_Acc  = val_dnls(net=net, loader_val=loader_val )
                    # Quantile_G()
                    if l2_Acc < Best_acc:
                        Best_acc = l2_Acc.copy()
                        best_model_net = copy.deepcopy(net.state_dict())
                        print('################## save model #################')
                        torch.save(net.state_dict(),'./DNLS-rep'+str(reps)+'.pth')
            iter_count += 1
    net.load_state_dict(best_model_net)
    return net

import torch
import json
import time
import shutil
import os
import sys
from tqdm import tqdm
from colorama import Fore, Style
from utils.utils import divide_tensors_into_lists, divide_tensors

# ASCII banner
ascii_banner_epoch = """
                                                                                
                          )                           (           )       (     
 (                     ( /(               )           )\   (   ( /(   (   )\ )  
 )\   `  )    (    (   )\())   (   (     (     `  )  ((_) ))\  )\()) ))\ (()/(  
((_)  /(/(    )\   )\ ((_)\    )\  )\    )\  ' /(/(   _  /((_)(_))/ /((_) ((_)) 
| __|((_)_\  ((_) ((_)| |(_)  ((_)((_) _((_)) ((_)_\ | |(_))  | |_ (_))   _| |  
| _| | '_ \)/ _ \/ _| | ' \  / _|/ _ \| '  \()| '_ \)| |/ -_) |  _|/ -_)/ _` |  
|___|| .__/ \___/\__| |_||_| \__|\___/|_|_|_| | .__/ |_|\___|  \__|\___|\__,_|  
     |_|                                      |_|                               
"""
# ASCII banner
ascii_banner_completed = """


                                                                                                                                        
                                                                                                                                        
                          .--.   _..._   .--.   _..._                                                                                   
                          |__| .'     '. |__| .'     '.   .--./)                                                                        
     .|  .-,.--.          .--..   .-.   ..--..   .-.   . /.''\\                                                                         
   .' |_ |  .-. |    __   |  ||  '   '  ||  ||  '   '  || |  | |                                                                        
 .'     || |  | | .:--.'. |  ||  |   |  ||  ||  |   |  | \`-' /                                                                         
'--.  .-'| |  | |/ |   \ ||  ||  |   |  ||  ||  |   |  | /("'`                                                                          
   |  |  | |  '- `" __ | ||  ||  |   |  ||  ||  |   |  | \ '---.                                                                        
   |  |  | |      .'.''| ||__||  |   |  ||__||  |   |  |  /'""'.\                                                                       
   |  '.'| |     / /   | |_   |  |   |  |    |  |   |  | ||     ||                                                                      
   |   / |_|     \ \._,\ '/   |  |   |  |    |  |   |  | \'. __//                                                                       
   `'-'           `--'  `"    '--'   '--'    '--'   '--'  `'---'                                                                        
       _..._       .-'''-.                                                                                                              
    .-'_..._''.   '   _    \                                         .---.                                                _______       
  .' .'      '.\/   /` '.   \  __  __   ___  _________   _...._      |   |      __.....__                    __.....__    \  ___ `'.    
 / .'          .   |     \  ' |  |/  `.'   `.\        |.'      '-.   |   |  .-''         '.              .-''         '.   ' |--.\  \   
. '            |   '      |  '|   .-.  .-.   '\        .'```'.    '. |   | /     .-''"'-.  `.      .|   /     .-''"'-.  `. | |    \  '  
| |            \    \     / / |  |  |  |  |  | \      |       \     \|   |/     /________\   \   .' |_ /     /________\   \| |     |  ' 
| |             `.   ` ..' /  |  |  |  |  |  |  |     |        |    ||   ||                  | .'     ||                  || |     |  | 
. '                '-...-'`   |  |  |  |  |  |  |      \      /    . |   |\    .-------------''--.  .-'\    .-------------'| |     ' .' 
 \ '.          .              |  |  |  |  |  |  |     |\`'-.-'   .'  |   | \    '-.____...---.   |  |   \    '-.____...---.| |___.' /'  
  '. `._____.-'/              |__|  |__|  |__|  |     | '-....-'`    |   |  `.             .'    |  |    `.             .'/_______.'/   
    `-.______ /                                .'     '.             '---'    `''-...... -'      |  '.'    `''-...... -'  \_______|/    
             `                               '-----------'                                       |   /                                  
                                                                                                 `'-'                                   

                                |_|                               
"""

def train_model(model, train_loader, val_loader, optimizer, scheduler, 
                loss_function, num_epochs, num_nodes, device, experiment_dir, 
                args, early_stop_patience, one_cells_weight=0.5):
    best_val_loss = float('inf')
    early_stop_counter = 0
    total_train_loss = []
    total_val_loss = []
    start_time = time.time()

    try:
        # ✅ Ensure directory exists
        os.makedirs(experiment_dir, exist_ok=True)

        # ✅ Optional: clear previous epoch_losses.json
        open(os.path.join(experiment_dir, 'epoch_losses.json'), 'w').close()

        # ✅ Initial loss before any training (Epoch 0)
        model.eval()
        initial_val_loss = validate_model(model, val_loader, loss_function, device, num_nodes, one_cells_weight)
        total_val_loss.append(initial_val_loss)
        total_train_loss.append(None)  # No training done yet
        print(f"Initial validation loss: {initial_val_loss}")

        # ✅ Save initial loss as epoch 0
        best_epoch = 0
        initial_epoch_data = {
            "epoch": 0,
            "train_loss": None,
            "val_loss": initial_val_loss,
            "best_epoch": best_epoch,
            "best_val_loss": initial_val_loss
        }
        with open(os.path.join(experiment_dir, 'epoch_losses.json'), 'a') as f:
            json.dump(initial_epoch_data, f)
            f.write('\n')
        print("✅ Epoch 0 saved to epoch_losses.json")
        
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0

            for cc in tqdm(train_loader):
                print(f"{Fore.GREEN}{Style.BRIGHT}At Epoch {epoch+1}:")
                x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, b10_t, b20_t = [tensor.to(device) for tensor in cc]

                # Convert adjacency matrices to Float
                a1 = a1.float()
                a2 = a2.float()
                coa2 = coa2.float()
                b1 = b1.float()
                b2 = b2.float()

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                sampled_b10, sampled_b20 = model(x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes)
                # print("ONE FORWARD PASS DONE")

                # Compute loss
                # loss_1 = loss_function(sampled_b10, b10_t)
                # loss_2 = loss_function(sampled_b20, b20_t)
                # loss = (one_cells_weight*loss_1 + (1-one_cells_weight)*loss_2)

                loss = loss_function(sampled_b10, b10_t)
                print("loss:", loss)
                # sys.exit()

                if isinstance(loss, torch.Tensor):
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                else:
                    print("Loss is not a tensor; skipping backward pass.")
            
                # print("ONE BACKWARD PASS DONE")

            avg_train_loss = epoch_train_loss / len(train_loader)
            total_train_loss.append(avg_train_loss)

            # Validation phase
            avg_val_loss = validate_model(model, val_loader, loss_function, device, num_nodes, one_cells_weight=0.5)
            total_val_loss.append(avg_val_loss)

            # Check if current model is the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict()  # Save the best model state
                early_stop_counter = 0
                # Save best model state to file
                torch.save(best_model_state, os.path.join(experiment_dir, 'best_val_model.pth'))
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered!")
                    break

            # Adjust learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save losses and model state after each epoch
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss
            }
            with open(os.path.join(experiment_dir, 'epoch_losses.json'), 'a') as f:
                json.dump(epoch_data, f)
                f.write('\n')

            print(ascii_banner_epoch)
            print(f"{Fore.GREEN}{Style.BRIGHT}Epoch {epoch+1}:")
            print(f"🔥 Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}{Style.RESET_ALL}")

            if early_stop_counter >= early_stop_patience:
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model and exiting...")
        save_training_state(total_train_loss, total_val_loss, start_time, experiment_dir, best_epoch, best_model_state)
        sys.exit()
    
    print(ascii_banner_completed)
    # Save training logs after normal completion
    save_training_state(total_train_loss, total_val_loss, start_time, experiment_dir, best_epoch, best_model_state)

    # Copy experiment_dir and rename it to last_experiment
    last_experiment_dir = os.path.join(args.save_dir, 'last_experiment')
    if os.path.exists(last_experiment_dir):
        shutil.rmtree(last_experiment_dir)  # Remove existing last_experiment directory if it exists
    shutil.copytree(experiment_dir, last_experiment_dir)
    print(f"Copied {experiment_dir} to {last_experiment_dir}")

def validate_model(model, val_loader, loss_function, device, num_nodes, one_cells_weight=0.5):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for cc in val_loader:
            x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, b10_t, b20_t = [tensor.to(device) for tensor in cc]

            # Convert adjacency matrices to Float
            a1 = a1.float()
            a2 = a2.float()
            coa2 = coa2.float()
            b1 = b1.float()
            b2 = b2.float()

            sampled_b10, sampled_b20 = model(x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes)
            # if not torch.all((sampled_b10 >= 0) & (sampled_b10 <= 1)):
            #     print("sampled_b10 contains values outside [0, 1]")
            #     print("Values below 0:", sampled_b10[sampled_b10 < 0])
            #     print("Values above 1:", sampled_b10[sampled_b10 > 1])
            # if not torch.all((b10_t >= 0) & (b10_t <= 1)):
            #     print("b10_t contains values outside [0, 1]")
            #     print("Values below 0:", b10_t[b10_t < 0])
            #     print("Values above 1:", b10_t[b10_t > 1])

            # Compute loss
            # loss_1 = loss_function(sampled_b10, b10_t)
            # loss_2 = loss_function(sampled_b20, b20_t)
            # loss = (one_cells_weight*loss_1 + (1-one_cells_weight)*loss_2)

            loss = loss_function(sampled_b10, b10_t)
            # print("loss:", loss)
            # sys.exit()

            # Check if the loss is a tensor (non-empty) or a float (empty tensors handled by returning 0.0)
            if isinstance(loss, torch.Tensor):
                total_val_loss += loss.item()
            else:
                total_val_loss += loss  # Here loss is already a float, no need to call .item()

    return total_val_loss / len(val_loader)

def save_training_state(train_loss, val_loss, start_time, experiment_dir, best_epoch, best_model_state):
    end_time = time.time()
    training_data = {
        "total_training_time": end_time - start_time,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_validation_epoch": best_epoch
    }
    with open(os.path.join(experiment_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f)
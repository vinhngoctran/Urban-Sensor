import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import argparse
import pickle
import scipy.io as sio
import h5py
import torch.cuda.amp as amp
import subprocess
import optuna
import joblib
import os
import pywt
import shap
import imageio
from captum.attr import IntegratedGradients
from tqdm import tqdm
import io

def check_cuda_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory
        print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
        print(f"Reserved GPU memory: {reserved_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated GPU memory: {allocated_memory / (1024 ** 3):.2f} GB")
        print(f"Free GPU memory: {free_memory / (1024 ** 3):.2f} GB")

def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
# Define the model classes (WaveletPreprocessing, PatchEmbedding, MultiHeadAttention, TransformerBlock, ViTPrediction)
class WaveletPreprocessing(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        coeffs = []
        for i in range(batch_size):
            for j in range(channels):
                coeff = pywt.wavedec2(x[i, j].cpu().numpy(), wavelet=self.wavelet, level=self.level)
                coeffs.append(coeff)
        reconstructed = []
        for coeff in coeffs:
            rec = pywt.waverec2(coeff, wavelet=self.wavelet)
            # Ensure the reconstructed image has the same dimensions as the input
            rec = rec[:height, :width]
            reconstructed.append(rec)
        return torch.tensor(np.array(reconstructed)).float().to(x.device).view(batch_size, channels, height, width)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x)[0]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTPrediction(nn.Module):
    def __init__(self, input_height, input_width, output_height, output_width, patch_size, in_channels, embed_dim, num_layers, 
                 num_heads, input_sequence_length, output_sequence_length, dropout_rate=0.1,use_wavelet=False):
        super().__init__()
        self.use_wavelet = use_wavelet
        if self.use_wavelet:
            self.wavelet_preprocessing = WaveletPreprocessing()
            print("2D Wavelet Transform is used")
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        num_patches = (input_height // patch_size) * (input_width // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_height * output_width)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.dropout = nn.Dropout(dropout_rate)
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, x):
        B, T, C, H, W = x.shape
        #x = x.contiguous().view(B * T, C, H, W)
        x = x.view(B * T, C, H, W)
        if self.use_wavelet:
            x = self.wavelet_preprocessing(x)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]  # Use only the CLS token for prediction
        x = self.head(x)
        x = x.view(B, T, self.output_height, self.output_width)
        x = x[:, :self.output_sequence_length, :, :]  # Select the first output_sequence_length time steps
        return x

def train_model(model, train_loader, val_loader, num_epochs, device, optimizer, early_stopping):
    criterion = nn.MSELoss()
    model = nn.DataParallel(model)  # Use DataParallel for multi-GPU training
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_combined, batch_inundation in train_loader:
            batch_combined, batch_inundation = batch_combined.to(device), batch_inundation.to(device)
            optimizer.zero_grad()
            outputs = model(batch_combined)
            #outputs = outputs.cpu()  # Move outputs to CPU
            loss = criterion(outputs, batch_inundation)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_combined, batch_inundation in val_loader:
                batch_combined, batch_inundation = batch_combined.to(device), batch_inundation.to(device)
                outputs = model(batch_combined)
                #outputs = outputs.cpu()  # Move outputs to CPU
                loss = criterion(outputs, batch_inundation)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def monte_carlo_dropout_predict(model, test_data, num_samples, device, batch_size=8):
    model.train()  # Set the model to training mode to keep dropout active
    predictions = []
    check_cuda_memory()
    # Process the test data in smaller batches
    for _ in range(num_samples):
        batch_predictions = []
        for i in range(0, test_data.size(0), batch_size):
            batch = test_data[i:i+batch_size].to(device)
            with torch.no_grad():
                pred = model(batch)
                batch_predictions.append(pred.cpu().numpy())
            check_cuda_memory()  # Check CUDA memory usage

        # Concatenate batch predictions
        predictions.append(np.concatenate(batch_predictions, axis=0))

    return np.array(predictions)

def input_perturbation_predict(model, test_data, num_samples, noise_std, device, batch_size=8):
    model.eval()
    predictions = []

    for _ in range(num_samples):
        batch_predictions = []
        for i in range(0, test_data.size(0), batch_size):
            batch = test_data[i:i+batch_size].to(device)
            perturbed_batch = batch + torch.randn_like(batch) * noise_std
            with torch.no_grad():
                pred = model(perturbed_batch)
                batch_predictions.append(pred.cpu().numpy())
            check_cuda_memory()  # Check CUDA memory usage

        # Concatenate batch predictions
        predictions.append(np.concatenate(batch_predictions, axis=0))

    return np.array(predictions)

def save_results(ensemble_predictions, mean_predictions, std_predictions):
    # Save to Python pickle format
    data_to_save = {
        'ensemble_predictions': ensemble_predictions,
        'mean_predictions': mean_predictions,
        'std_predictions': std_predictions
    }
    with open(args.ProjectName + '_prediction_results.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print("Results saved in Python pickle format.")

    # Save to MATLAB format
    sio.savemat(args.ProjectName +'_prediction_results.mat', data_to_save)
    print("Results saved in MATLAB format.")
    
def load_data_train(device):
    
    data = sio.loadmat(args.train_data_paths)

    
    rainfall_data = data['X1'].astype('float64')
    rainfall_data = np.transpose(rainfall_data, (3, 2, 0, 1))
    
    Static_data = data['X2'].astype('float64')
    Static_data = np.transpose(Static_data, (3, 2, 0, 1))
    
    inundation_data = data['Y'].astype('float64')
    inundation_data = np.transpose(inundation_data, (3, 2, 0, 1))
    
    input_height = rainfall_data.shape[2]
    input_width = rainfall_data.shape[3]
    output_height = inundation_data.shape[2]
    output_width = inundation_data.shape[3]
    in_channels = 2
    input_sequence_length = rainfall_data.shape[1]
    output_sequence_length = inundation_data.shape[1]
    
    # Training data: 100 sinulations
    combined_data = np.stack((rainfall_data, Static_data), axis=2)  
    combined_tensor = torch.FloatTensor(combined_data).to(device)
    inundation_tensor = torch.FloatTensor(inundation_data).to(device)
    training_data = TensorDataset(combined_tensor, inundation_tensor)  
    
    return training_data,input_height,input_width,output_height,output_width,in_channels,input_sequence_length,output_sequence_length

def load_data_val(device):
    
    data = sio.loadmat(args.val_data_paths)
    
    rainfall_data = data['X1'].astype('float64')
    rainfall_data = np.transpose(rainfall_data, (3, 2, 0, 1))
    
    Static_data = data['X2'].astype('float64')
    Static_data = np.transpose(Static_data, (3, 2, 0, 1))
    
    inundation_data = data['Y'].astype('float64')
    inundation_data = np.transpose(inundation_data, (3, 2, 0, 1))
    

    # Training data: 100 sinulations
    combined_data = np.stack((rainfall_data, Static_data), axis=2)  
    combined_tensor = torch.FloatTensor(combined_data).to(device)
    inundation_tensor = torch.FloatTensor(inundation_data).to(device)
    Val_data = TensorDataset(combined_tensor, inundation_tensor)  
    
    return Val_data

def load_data_test(device):
    
    data = sio.loadmat(args.test_data_paths)

    
    rainfall_data = data['X1'].astype('float64')
    rainfall_data = np.transpose(rainfall_data, (3, 2, 0, 1))
    
    Static_data = data['X2'].astype('float64')
    Static_data = np.transpose(Static_data, (3, 2, 0, 1))
    
    inundation_data = data['Y'].astype('float64')
    inundation_data = np.transpose(inundation_data, (3, 2, 0, 1))
    
    # Testing data: 100 simulation
    testing_data = np.stack((rainfall_data, Static_data), axis=2) 
    
    return testing_data

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch_combined, batch_inundation in dataloader:
            batch_combined, batch_inundation = batch_combined.to(device), batch_inundation.to(device)
            outputs = model(batch_combined)
            #outputs = outputs.cpu()  # Move outputs to CPU
            loss = criterion(outputs, batch_inundation)
            total_loss += loss.item()
    return total_loss / len(dataloader)

        
def objective(trial):
    #dataloader,input_height,input_width,output_height,output_width,in_channels,input_sequence_length,output_sequence_length,device
    # Define the hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # Ensure embed_dim is divisible by num_heads
    num_heads = trial.suggest_int('num_heads', 4, 16)
    embed_dim = trial.suggest_int('embed_dim', 128, 512, step=16)
    embed_dim = embed_dim - (embed_dim % num_heads)  # Make sure embed_dim is divisible by num_heads
    num_layers = trial.suggest_int('num_layers', 2, 8)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    criterion = nn.MSELoss()
    
    # Ensure embed_dim is divisible by num_heads
    if embed_dim % num_heads != 0:
        embed_dim = (embed_dim // num_heads) * num_heads
        
    # Create the model with the suggested hyperparameters  
    model = ViTPrediction(
        input_height=args.input_height,
        input_width=args.input_width,
        output_height=args.output_height,
        output_width=args.output_width,
        patch_size=8,
        in_channels=args.in_channels,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        input_sequence_length=args.input_sequence_length,
        output_sequence_length=args.output_sequence_length,
        dropout_rate=dropout_rate,
        use_wavelet=args.use_wavelet
    ).to(args.device)

    # Train the model
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_model(model, args.dataloader,args.val_loader , args.num_epochs, args.device, optimizer,early_stopping)

    # Evaluate the model on the validation set
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_combined, batch_inundation in args.val_loader:
            batch_combined, batch_inundation = batch_combined.to(args.device), batch_inundation.to(args.device)
            outputs = model(batch_combined)
            #outputs = outputs.cpu()  # Move outputs to CPU
            loss = criterion(outputs, batch_inundation)
            val_loss += loss.item()
    
    val_loss /= len(args.val_loader)
    
    # Clear CUDA memory
    clear_cuda_memory()
    return val_loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
## Explainable AI
class ModelWrapper(nn.Module):
    def __init__(self, model, output_index, target_location):
        super().__init__()
        self.model = model
        self.output_index = output_index
        self.target_location = target_location

    def forward(self, x):
        outputs = self.model(x)
        output = outputs[self.output_index]
        
        # Check the shape of the output and adjust indexing accordingly
        if output.dim() == 3:  # (batch, height, width)
            return output[:, self.target_location[0], self.target_location[1]]
        elif output.dim() == 4:  # (batch, time, height, width)
            return output[:, :, self.target_location[0], self.target_location[1]]
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")


def integrated_gradients_explanation(model, input_data, target_location, output_index, n_steps=50):
    model.eval()
    wrapped_model = ModelWrapper(model, output_index, target_location)
    ig = IntegratedGradients(wrapped_model)
    
    attributions = ig.attribute(input_data, n_steps=n_steps, internal_batch_size=1)
    return attributions

def create_attribution_gif(attributions, target_location):
    images = []
    for t in range(attributions.shape[1]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, channel_name in enumerate(['Rainfall', 'Temperature']):
            attr_data = attributions[0, t, i].cpu().numpy()
            vmax = np.abs(attr_data).max()
            im = axes[i].imshow(attr_data, cmap='RdBu', vmin=-vmax, vmax=vmax)
            axes[i].set_title(f'{channel_name} Attribution (t={t+1})')
            plt.colorbar(im, ax=axes[i])
            
            axes[i].plot(target_location[1], target_location[0], 'r*', markersize=10)
            axes[i].text(target_location[1], target_location[0], 'Target', color='red', 
                         ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close(fig)
    
    # Save as GIF
    save_path=args.ProjectName+ '_attributions.gif'
    imageio.mimsave(save_path, images, duration=500)  # 500ms per frame
    print(f"Attribution GIF saved to {save_path}")



def load_results(file_path):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results

def load_data2():
    
    data = sio.loadmat(args.test_data_paths)

    inundation_data = data['Y'].astype('float64')
    inundation_data = np.transpose(inundation_data, (3, 2, 0, 1))

    return inundation_data

def plot_comparison(results, reference_data):
    mean_prediction = results['mean_predictions']
    
    vmin = min(np.min(mean_prediction[1]), np.min(reference_data[1]))
    vmax = max(np.max(mean_prediction[1]), np.max(reference_data[1]))
    
    fig, axes = plt.subplots(6, 8, figsize=(20, 25))
    fig.suptitle('Predicted vs Observed', fontsize=16)
    
    for i in range(reference_data.shape[1]):
        ax1 = axes[i // 4, (i % 4) * 2]
        ax2 = axes[i // 4, (i % 4) * 2 + 1]
        
        im1 = ax1.imshow(np.flipud(mean_prediction[1,i,:,:]), cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'AI (t={i+1})')
        ax1.axis('off')
        if i==0:
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
        ax2.imshow(np.flipud(reference_data[1,i,:,:]), cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title('tRIBS (t={i+1})')
        ax2.axis('off')
        
        # Add a single colorbar for the entire figure
    #cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
   # cbar.set_label('Value')
    
    plt.tight_layout()
    plt.savefig(args.ProjectName+ '_PredictedvsObs.jpg', dpi=300, bbox_inches='tight')
    print("Comparison plots saved.")
    
    
    images = []
    for t in range(24):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        

        im = axes[0].imshow(np.flipud(mean_prediction[1,t,:,:]), cmap='Greys', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'AI (t={t+1})')
        plt.colorbar(im, ax=axes[0])
        
        im = axes[1].imshow(np.flipud(reference_data[1,t,:,:]), cmap='Greys', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'tRIBS (t={t+1})')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close(fig)
    
    # Save as GIF
    save_path=args.ProjectName+ '_Predic_.gif'
    imageio.mimsave(save_path, images, duration=500)  # 500ms per frame
    print(f"Attribution GIF saved to {save_path}")

    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    check_cuda_memory()
    # Prepare your data
    dataset,input_height,input_width,output_height,output_width,in_channels,input_sequence_length,output_sequence_length =  load_data_train(device)
    val_data = load_data_val(device)
    testing_data = load_data_test(device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    
    args.device = device
    args.dataloader = dataloader
    args.val_loader = val_loader
    args.input_height = input_height
    args.input_width = input_width
    args.output_height = output_height
    args.output_width = output_width
    args.in_channels = in_channels
    args.input_sequence_length = input_sequence_length
    args.output_sequence_length = output_sequence_length
    
    if args.BoA:
        # Check if best hyperparameters file exists
        best_hyperparams_path = args.ProjectName+'_best_hyperparameters.pkl'
        if os.path.exists(best_hyperparams_path):
            # Load best hyperparameters
            best_params = joblib.load(best_hyperparams_path)
            print("Loaded best hyperparameters from file.")
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=args.trial)  # Adjust n_trials as needed
            clear_cuda_memory()
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                
            # Save the best hyperparameters
            best_params = trial.params
            joblib.dump(best_params, best_hyperparams_path)
            print("Best hyperparameters saved to 'best_hyperparameters_ViT.pkl'")
        
        # Ensure embed_dim is divisible by num_heads
        if best_params['embed_dim'] % best_params['num_heads'] != 0:
            best_params['embed_dim'] = (best_params['embed_dim'] // best_params['num_heads']) * best_params['num_heads']
            
        best_model = ViTPrediction(
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
            patch_size=8,
            in_channels=in_channels,
            embed_dim=best_params['embed_dim'],
            num_layers=best_params['num_layers'],
            num_heads=best_params['num_heads'],
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            dropout_rate=best_params['dropout_rate'],
            use_wavelet=args.use_wavelet
        ).to(device)
        
    else:
        best_model = ViTPrediction(
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
            patch_size=8,
            in_channels=in_channels,
            embed_dim=args.s_embed_dim,
            num_layers=args.s_num_layers,
            num_heads=args.s_num_heads,
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            dropout_rate= args.s_dropout_rate,
            use_wavelet=args.use_wavelet
        ).to(device)

    # Check if a saved model exists
    model_path = args.ProjectName +'_best_model.pth'
    if os.path.exists(model_path):
        print("Loading saved model...")
        best_model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    else:
        if args.BoA:
            best_optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
        else:
            best_optimizer = optim.Adam(best_model.parameters(), lr=args.s_lr)
        train_start_time = time.time()
        early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
        train_model(best_model, dataloader, val_loader,args.num_epochs, device, best_optimizer,early_stopping)
        train_end_time = time.time()
        total_train_time = train_end_time - train_start_time
        print(f"Total training time: {total_train_time:.2f} seconds")
        # Save the best model
        torch.save(best_model.state_dict(), model_path)
        print("Best model saved to 'best_model_ViT.pth'")
    
    test_combined_tensor = torch.FloatTensor(testing_data).to(device)

    if args.PredicMod:
        predict_start_time = time.time()
            
        if args.ensemble_method == 'mc_dropout':
            ensemble_predictions = monte_carlo_dropout_predict(best_model, test_combined_tensor, args.num_samples, device, batch_size=args.batch_size)
        elif args.ensemble_method == 'input_perturbation':
            ensemble_predictions = input_perturbation_predict(best_model, test_combined_tensor, args.num_samples, args.noise_std, batch_size=args.batch_size)
        else:
            raise ValueError("Invalid ensemble method. Choose 'mc_dropout' or 'input_perturbation'.")
    
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        std_predictions = np.std(ensemble_predictions, axis=0)
    
        predict_end_time = time.time()
        predict_time = predict_end_time - predict_start_time
        print(f"Prediction time: {predict_time:.2f} seconds")
    
        
        # Save results
        save_results(ensemble_predictions, mean_predictions, std_predictions)
    
    
    if args.XAI:
        target_location = (16, 32)  # Example target location
        test_input = test_combined_tensor[:1]  # Use the first sample for explanation
        attributions = integrated_gradients_explanation(best_model, test_input, target_location, output_index=0)
        save_path=args.ProjectName +'_ExplainableAI.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(attributions, f)
        print(f"SA values saved to {save_path}")
        create_attribution_gif(attributions, target_location)
        print("Integrated Gradients explanation completed.")

    
    if args.Visualize:
        results = load_results(args.ProjectName+'_prediction_results.pkl')
        reference_data = load_data2()
        plot_comparison(results, reference_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inundation Prediction with ViT")
    parser.add_argument("--ensemble_method", type=str, choices=['mc_dropout', 'input_perturbation'], default='mc_dropout', help="Ensemble method to use")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples for ensemble prediction")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Standard deviation of noise for input perturbation")
    
    #Hyper-parameters option
    parser.add_argument("--BoA", type=bool, default=True, help="Use 2D wavelet transform preprocessing")
    parser.add_argument("--trial", type=int, default=30, help="Number of BoA trial")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--s_embed_dim", type=int, default=256, help="Embedding dimensions")
    parser.add_argument("--s_num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--s_num_heads", type=int, default=8, help="Number of headings")
    parser.add_argument("--s_dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--s_lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    
    
    #File name
    parser.add_argument('--train_data_paths', type=str, default='Input/Data_Train_I.mat')
    parser.add_argument('--val_data_paths', type=str, default='Input/Data_Val_I.mat')
    parser.add_argument('--test_data_paths', type=str, default='Input/Data_Test_I.mat')
    parser.add_argument('--ProjectName', type=str, default='results/DA')
    
    # Model option
    parser.add_argument("--use_wavelet", type=bool, default=False, help="Use 2D wavelet transform preprocessing")
    parser.add_argument("--PredicMod", type=bool, default=True, help="Use 2D wavelet transform preprocessing")
    parser.add_argument("--XAI", type=bool, default=True, help="Explainable AI option")
    parser.add_argument("--Visualize", type=bool, default=True, help="Visualization option")

    args = parser.parse_args()

    main(args)
















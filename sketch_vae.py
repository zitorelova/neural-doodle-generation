import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.autograd.variable import Variable
import torch.nn.functional as F
from skimage.util import montage

import warnings
warnings.filterwarnings('ignore')

gpu = torch.cuda.is_available()

class Hps():
    """
    Default hyperparameters for training the model
    
    """
    def __init__(self):
        checks = [f for f in os.listdir('data') if f.endswith('.npz')]
        if not checks:
            raise FileNotFoundError("No .npz files found in data folder")
        else: 
            data_location = 'data/{}'.format(checks[0])
        self.data_location = data_location
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200

hp = Hps()

def clean_strokes(strokes):
    """
    Remove stroke sequences that are too long or too short

    Arguments:
    strokes (np.array): Sequence of strokes to clean
    
    """
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def max_size(data):
    """
    Get longest sequence length

    Arguments:
    data (np.array): Array of stroke sequences

    """
    sizes = [len(seq) for seq in data]
    return max(sizes)

def calculate_normalizing_scale_factor(strokes):
    """
    Calculate normalizing scale factor as explained in section 1 (Dataset Details) of 
    the supplementary material for A Neural Representation of Sketch Drawings
   
    Arguments:
    strokes (np.array): Array of strokes to normalize

    """
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """
    Normalize the entire dataset (delta_x, delta_y) by the scaling factor

    Arguments:
    strokes (np.array): Array of strokes to normalize
    
    """
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data

dataset = np.load(hp.data_location, encoding='latin1')
data = dataset['train']
data = clean_strokes(data)
data = normalize(data)
Nmax = max_size(data)

def make_batch(batch_size):
    """
    Create batch for model training

    Arguments:
    batch_size (int): Size of batch for training

    """
    batch_idx = np.random.choice(len(data),batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:,0])
        new_seq = np.zeros((Nmax,5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0
        lengths.append(len(seq[:,0]))
        strokes.append(new_seq)
        indice += 1

    if gpu:
        batch = Variable(torch.from_numpy(np.stack(strokes,1)).cuda().float())
    else:
        batch = Variable(torch.from_numpy(np.stack(strokes,1)).float())
    return batch, lengths

def lr_decay(optimizer):
    """
    Decay learning rate by a factor of lr_decay

    Arguments:
    optimizer (torch.optim.Optimizer): Pytorch optimizer to decay
    
    """
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

class EncoderRNN(nn.Module):
    """
    Encoder class for the model

    """
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, \
            dropout=hp.dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        """
        Forward pass through encoder
        
        Arguments:
        inputs (torch.Tensor): Inputs to the encoder model
        batch_size (int): Size of batch for model training
        hidden_cell (torch.Tensor): Hidden layer for encoder model

        """
        if hidden_cell is None:
            if gpu: 
                hidden = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size, hp.enc_hidden_size)
                cell = torch.zeros(2, batch_size, hp.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden,cell) = self.lstm(inputs.float(), hidden_cell)
        hidden_forward, hidden_backward = torch.split(hidden,1,0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        z_size = mu.size()
        if gpu:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size))
        z = mu + sigma*N
        return z, mu, sigma_hat

class DecoderRNN(nn.Module):
    """
    Decoder class for the model

    """
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.fc_hc = nn.Linear(hp.Nz, 2*hp.dec_hidden_size)
        self.lstm = nn.LSTM(hp.Nz+5, hp.dec_hidden_size, dropout=hp.dropout)
        self.fc_params = nn.Linear(hp.dec_hidden_size,6*hp.M+3)

    def forward(self, inputs, z, hidden_cell=None):
        """
        Forward pass through decoder

        Arguments:
        inputs (torch.Tensor): Inputs to the decoder model
        z (torch.Tensor): Vector z constructed from outputs of encoder model
        hidden_cell (torch.Tensor): Hidden layer for decoder model

        """

        if hidden_cell is None:
            hidden,cell = torch.split(F.tanh(self.fc_hc(z)),hp.dec_hidden_size,1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1]) 
        params_pen = params[-1] 
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2)
        if self.training:
            len_out = Nmax+1
        else:
            len_out = 1
                                   
        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,hp.M)
        q = F.softmax(params_pen).view(len_out,-1,3)
        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,q,hidden,cell

class Model():
    """
    Full VAE model (Encoder + Decoder)
    
    """
    def __init__(self):
        if gpu:
            self.encoder = EncoderRNN().cuda()
            self.decoder = DecoderRNN().cuda()
        else:
            self.encoder = EncoderRNN()
            self.decoder = DecoderRNN()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def make_target(self, batch, lengths):
        """
        Create targets for the model

        Arguments:
        batch (torch.Tensor): Batch to create targets from
        lengths (list): lengths of each of the inputs

        """
        if gpu:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.size()[1]).cuda().unsqueeze(0)
        else:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(Nmax+1, batch.size()[1])
        for indice,length in enumerate(lengths):
            mask[:length,indice] = 1
        if gpu:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:,:,0]]*hp.M,2)
        dy = torch.stack([batch.data[:,:,1]]*hp.M,2)
        p1 = batch.data[:,:,2]
        p2 = batch.data[:,:,3]
        p3 = batch.data[:,:,4]
        p = torch.stack([p1,p2,p3],2)
        return mask,dx,dy,p

    def train(self, iteration):
        """
        Function for training the model

        Arguments:
        iteration (int): The current iteration number

        """
        self.encoder.train()
        self.decoder.train()
        iteration += 1
        batch, lengths = make_batch(hp.batch_size)
        z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)
        if gpu:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).unsqueeze(0)
        batch_init = torch.cat([sos, batch],0)
        z_stack = torch.stack([z]*(Nmax+1))
        inputs = torch.cat([batch_init, z_stack],2)
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        mask,dx,dy,p = self.make_target(batch, lengths)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.eta_step = 1-(1-hp.eta_min)*hp.R
        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask,dx,dy,p)
        loss = LR + LKL
        loss.backward()
        nn.utils.clip_grad_norm(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(), hp.grad_clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        if not iteration % 1:
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)
        if not iteration % 200:
            print(f'Iteration: {iteration}\n{"-" * 30}\nFull loss: {loss.item() :.3f}\nReconstruction loss: {LR.item() :.3f}\nKL loss: {LKL.item() :.3f}\n')
            self.save(iteration)
#            self.conditional_generation(iteration)

    def bivariate_normal_pdf(self, dx, dy):
        """
        Bivariate normal pdf modeled from GMM with N normal distributions

        Arguments:
        dx (torch.Tensor): Delta x offset term to parameterize the bivariate normal distribution 
        dy (torch.Tensor): Delta y offset term to parameterize the bivariate normal distribution

        """
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def reconstruction_loss(self, mask, dx, dy, p):
        """
        Reconstruction loss to be used as criterion for the model

        Arguments:
        mask (torch.Tensor): Mask for LS portion of reconstruction loss
        dx (torch.Tensor): Delta x that parameterizes the bivariate normal distribution
        dy (torch.Tensor): Delta y that parameterizes the bivariate normal distribution
        p  (torch.Tensor): Pen state terms for LP portion of reconstruction loss
        """
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask*torch.log(1e-5+torch.sum(self.pi * pdf, 2)))\
            /float(Nmax*hp.batch_size)
        LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS+LP

    def kullback_leibler_loss(self):
        """
        Kullback-Leibler loss to be used as criterion for the model

        """
        LKL = -0.5*torch.sum(1+self.sigma-self.mu**2-torch.exp(self.sigma))\
            /float(hp.Nz*hp.batch_size)
        if gpu:
            KL_min = Variable(torch.Tensor([hp.KL_min]).cuda()).detach()
        else:
            KL_min = Variable(torch.Tensor([hp.KL_min])).detach()
        return hp.wKL*self.eta_step * torch.max(LKL,KL_min)

    def save(self, iteration):
        """
        Save state dict of the model

        Arguments:
        iteration (int): Iteration number

        """
        torch.save(self.encoder.state_dict(), 'checkpoints/encoderRNN_iter_{}.pth'.format(iteration))
        torch.save(self.decoder.state_dict(), 'checkpoints/decoderRNN_iter_{}.pth'.format(iteration))
        
    def load(self, encoder_name, decoder_name):
        """

        Load in saved model from .pth file

        Arguments:
        encoder_name (str): Path to the saved encoder weights
        decoder_name (str): Path to the saved decoder weights

        """
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self, iteration):
        """
        Generate image from the model

        Arguments:
        iteration (int): Iteration number

        """
        batch,lengths = make_batch(1)
        self.encoder.train(False)
        self.decoder.train(False)
        z, _, _ = self.encoder(batch, 1)
        if gpu:
            sos = Variable(torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda())
        else:
            sos = Variable(torch.Tensor([0,0,1,0,0]).view(1,1,-1))
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(Nmax):
            input = torch.cat([s,z.unsqueeze(0)],2)
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            s, dx, dy, pen_down, eos = self.sample_next_state()
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                break
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample,y_sample,z_sample]).T
        make_image(sequence, iteration)

    def sample_next_state(self):

        def adjust_temp(pi_pdf):
            """
            Adjust temperature to control randomness

            Arguments:
            pi_pdf (torch.Tensor): Probability density function 

            """
            pi_pdf = np.log(pi_pdf)/hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        q = self.q.data[0,0,:].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        mu_x = self.mu_x.data[0,0,pi_idx]
        mu_y = self.mu_y.data[0,0,pi_idx]
        sigma_x = self.sigma_x.data[0,0,pi_idx]
        sigma_y = self.sigma_y.data[0,0,pi_idx]
        rho_xy = self.rho_xy.data[0,0,pi_idx]
        x,y = sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        if gpu:
            return Variable(next_state.cuda()).view(1,1,-1),x,y,q_idx==1,q_idx==2
        else:
            return Variable(next_state).view(1,1,-1),x,y,q_idx==1,q_idx==2

def sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
    """
    Sample from bivariate normal parameterized by outputs from encoder

    Arguments:
    mu_x (torch.Tensor): Mean x for parameterizing bivariate normal distribution
    mu_y (torch.Tensor): Mean y for parameterizing bivariate normal distribution
    sigma_x (torch.Tensor): Standard deviation x for parameterizing bivariate normal distribution
    sigma_y (torch.Tensor): Standard deviation y for parameterizing bivariate normal distribution
    rho_xy (torch.Tensor): Correlation parameter for bivariate normal distribution
    greedy (boolean): Whether to randomly sample from distribution

    """
    if greedy:
      return mu_x,mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def make_image(sequence, iteration, name='generated_'):
    """
    Plot strokes as image and save as JPEG

    Arguments:
    sequence (np.array): Numpy array of strokes from conditional generation
    iteration (int): Iteration number
    name (str): Prefix to use when saving generated image

    """ 
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.axis('off')
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())
    name = 'assets/' + name + str(iteration) + '.jpg'
    pil_image.save(name,"JPEG")
    plt.close("all")

# Deprecated
def stitch_images_old(directory='assets', width=480, length=640):
    """
    Stitch generated images together in a grid

    Arguments:
    directory (str): Directory where generated images are located
    width (int): Width of images
    length (int): Length of images
  
    """
    img_paths = [f for f in os.listdir(directory) if 'generated' in f]
    grid = np.zeros((width * int(len(img_paths) ** 0.5), length * int(len(img_paths) ** 0.5), 3))
    lat, lon = 0, 0
    for img in img_paths:
        if lat == grid.shape[0]:
            lat = 0
            lon += length
        grid[lat: lat + width, lon: lon + length, :] = plt.imread(os.path.join(directory, img))
        lat += width
    return grid
     
def stitch_images(directory='assets'):
    """
    Stitch generated images together in a grid

    Arguments:
    directory (str): Directory where generated images are located

    """
    img_paths = [f for f in os.listdir(directory) if 'generated' in f]
    assert len(img_paths) == 9
    raw_arr = [plt.imread(os.path.join(directory, im)) for im in img_paths]
    raw_arr = np.stack(raw_arr, axis=0)
    stitched = montage(raw_arr, grid_shape=(3, 3), multichannel=True)
    cv2.imwrite(os.path.join(directory, 'stitched_img.jpg'), stitched)    

if __name__=="__main__":

    model = Model()
    iters = 4000
    print("Starting training run...\n")
    for iteration in range(iters):
        model.train(iteration)
    model.load('checkpoints/encoderRNN_iter_{}.pth'.format(iters),'checkpoints/decoderRNN_iter_{}.pth'.format(iters))
    print("Generating images...\n")
    for i in range(9):
        model.conditional_generation(i)
    stitch_images()


from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
from net.channel_est import UNetD
from net.channel_uce import Channel_Estimator

class LCFSC(nn.Module):
    def __init__(self, args, config):
        super(LCFSC, self).__init__()
        self.config = config
        
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        
        # self.ce = UNetD(in_chn = 2)
        self.ce = Channel_Estimator()
        
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model
        self.sig = nn.Sigmoid()

        self.cvae = CVAE(128*3, 648, 128)

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def LS_CE(self, s, y):
        h_LS = torch.matmul(y, torch.linalg.inv(s))

        return h_LS

    def stream_demapping(self, input, B):
        input = input.reshape(B, -1)
        return input

    def stream_mapping(self, input, Ntx, B):
        input = input.reshape(B, Ntx, -1)
        return input

    def forward(self, input_image, H_MIMO, given_SNR, training_epoch, tra):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR
        
        r = torch.randint(H_MIMO.shape[2], (1,))
        h = H_MIMO[:, :, r]
        h = h.squeeze(-1)
        h = h.cuda()
        
        s = torch.Tensor([[1,1,1,1,1,1,1,1], [1,-1,1,1,1,1,1,1], [1,1,-1,1,1,1,1,1], [1,1,1,-1,1,1,1,1], [1,1,1,1,-1,1,1,1], [1,1,1,1,1,-1,1,1], [1,1,1,1,1,1,-1,1], [1,1,1,1,1,1,1,-1]])
        
        s = s.unsqueeze(0) # pilots
        s = np.complex64(s)
        s = torch.from_numpy(s)
        s = s.cuda()

        receive_s = self.feature_pass_channel(s, chan_param, h)

        # LS或者NB_CE
        h_LS = self.LS_CE(s.squeeze(0), receive_s) # [1, 8, 8]
        h_LS = h_LS.unsqueeze(0) 
        
        h_LS = torch.concat((h_LS.real, h_LS.imag), dim = 1) # [1, 2, 8, 8]

        h_est = self.ce(h_LS)

        h_est = h_est[:,0,:,:] + 1j * h_est[:,1,:,:] # [1, 1, 8, 8]
        h_est = h_est.squeeze(0)

        initial_y = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]).cuda()
        initial_y = initial_y.to(torch.float32)
                
        with torch.no_grad():
            x_prior = self.encoder(input_image, chan_param, self.model, h_est, initial_y, tra)
        '''
        if training_epoch <= 1000:
            x_hat, condition_loc, condition_scale, prior_loc, prior_scale, y = self.cvae(x_prior.view(B, -1), initial_y)
        
        else:
            with torch.no_grad():
                x_hat, condition_loc, condition_scale, prior_loc, prior_scale, y = self.cvae(x_prior.view(B, -1), initial_y)
        '''
        with torch.no_grad():
                x_hat, condition_loc, condition_scale, prior_loc, prior_scale, y = self.cvae(x_prior.view(B, -1), initial_y)
        y = initial_y
        

        feature = self.encoder(input_image, chan_param, self.model, h_est, y, tra)

        B = feature.shape[0]
        feature_shape = feature.shape
        feature = feature.reshape(B, -1)      

        CBR = feature.numel() / input_image.numel() / 2

        # Feature pass channel
        L = feature.shape[1]
        feature = feature[:, :L // 2] + feature[:, L // 2:] * 1j

        feature = self.stream_mapping(feature, 8, B)
        noisy_feature = self.feature_pass_channel(feature, chan_param, h)
        noisy_feature = self.stream_demapping(noisy_feature, B)
        noisy_feature = torch.cat([torch.real(noisy_feature), torch.imag(noisy_feature)], dim = -1)
        noisy_feature = noisy_feature.reshape(feature_shape)
        recon_image = self.decoder(noisy_feature, chan_param, self.model, h)

        mse = self.squared_difference(input_image*255., recon_image.clamp(0.,1.)*255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0.,1.))
        res_image = input_image - recon_image

        loss_F  = self.squared_difference(x_prior, x_hat)
        loss_KL = - 0.5 * (1 - (prior_loc - condition_loc) ** 2/torch.exp(condition_scale) + prior_scale - condition_scale - torch.exp(prior_scale - condition_scale))
        
        return recon_image, res_image, CBR, chan_param, mse.mean(), loss_G.mean(), loss_F.mean(), loss_KL.mean()

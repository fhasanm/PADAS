import torch
from model import highwayNet


def call_model():
    args = initialize_args()
    model = highwayNet(args)
    model.load_state_dict(torch.load('trained_models/cslstm_m_7.pt'))
    # model = highwayNet(args)
    return model

def initialize_args():
    args = {}
    args['use_cuda'] = False
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = False
    args['train_flag'] = False
    return args

def traj_pred(model, tracks):


    with torch.no_grad():
        #for each vehicle
        # preds = torch.zeros(25, 2, tracks.shape[3])
        tracks = torch.from_numpy(tracks)
        # tracks = tracks.cuda()

        # z = tracks[:,2,:]
        # z = z.reshape(z.shape[0], z.shape[1], 1)
        tracks = tracks[:,:2,:]

        # change the coordinate frame
        tracks_ij = tracks
        tracks[:,0,:] = tracks_ij[:,1,:]
        tracks[:,1,:] = tracks_ij[:,0,:]

        tracks = tracks.reshape(1, tracks.shape[0], tracks.shape[1], tracks.shape[2]) # t c n -> b t c n
        tracks = tracks.permute(0,2,1,3) #b t c n -> b c t n

        predictions = torch.zeros(25,2,tracks.shape[3]) # change


        for i in range(tracks.shape[3]):

            # proj = tracks[:,:,-1,i].repeat(tracks.shape[3], 1)
            # assert proj.shape == tracks.shape, "proj and tracks mismatch"
            transform = tracks[:,:,0,i].reshape(tracks.shape[0],tracks.shape[1],1,1)
            transformed_tracks = tracks - transform # transforms the coordinate frame to target vehicle at t=0

            nbrs = transformed_tracks.reshape(transformed_tracks.shape[1],transformed_tracks.shape[2], transformed_tracks.shape[3])
            nbrs = nbrs.permute(1,2,0) #c,t,n -> t,n,c
            assert nbrs.shape == (tracks.shape[2], tracks.shape[3], tracks.shape[1]), "nbrs shape error"

            hist = transformed_tracks[:,:,:,i].reshape(transformed_tracks.shape[0],transformed_tracks.shape[1], transformed_tracks.shape[2])
            hist = hist.permute(2,0,1) #b,c,t -> t, b, c
            assert hist.shape == (tracks.shape[2], tracks.shape[0], tracks.shape[1]), "hist shape error"

            masks = torch.zeros([hist.shape[1], 3, 13, 64], device="cuda" if torch.cuda.is_available() and
                                                                             tracks.device=='cuda' else 'cpu').bool()
            fut = model(hist, nbrs, masks)
            fut = fut[:,:,:2]

            fut = fut.reshape(1,fut.shape[0], fut.shape[1], fut.shape[2]) #t b c -> 1 t b c
            fut = fut.permute(2,3,1,0) #n t b c -> b c t n
            fut = fut + transform
            fut = fut.reshape(fut.shape[1], fut.shape[2], fut.shape[3])
            fut = fut.permute(1, 0, 2) #c t n -> t c n
            fut = fut.reshape(fut.shape[0], fut.shape[1])

            #change x and y



            predictions[:,:,i] = fut

        # predictions_wz = torch.zeros(15, 3, predictions.shape[2])
        # predictions_wz[:,:2,:] = predictions[:15,:,:]
        # predictions_wz[:,2,:] = z
        #
        # predictions = predictions_wz

        #predictions = torch.cat((predictions[:15,:,:], z), dim=1)
        predictions = predictions[:15]
        return predictions.detach().numpy()

        # preds = predictions
        # preds = preds.detach().numpy()





if __name__ == '__main__':

    args = {}
    args['use_cuda'] = True
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = False
    args['train_flag'] = False

    model = highwayNet(args)
    model.load_state_dict(torch.load('trained_models/cslstm_m_0.pt'))
    model = highwayNet(args)
    tracks = torch.randn([2, 16, 120])

    traj_pred(model, tracks)
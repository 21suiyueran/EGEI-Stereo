import torch
from torch.utils.data import Dataset
import os
import numpy
from utils import filename_templates as TEMPLATES
from loader.utils import *
from utils.transformers import *
from utils import dataset_constants


class Mvsec_Dataset(Dataset):
    def __init__(self, args, type, path):
        super(Mvsec_Dataset, self).__init__() 

        self.path_dataset = path
        self.timestamp_files = {}
        self.timestamp_files_flow = {}
        # If we load the image timestamps, we consider the framerate to be 45Hz.
        # Else if we load the depth/flow timestamps, the framerate is 20Hz.
        # The update rate gets set to 20 or 40 in the "get indices" method
        self.update_rate = None
        self.dataset = self.get_indices(path, args['filter'], args['align_to'])  
        self.input_type = 'events'
        self.type = type # Train/Val/Test


        # Evaluation Type.  Dense  -> Valid where GT exists
        #                   Sparse -> Valid where GT & Events exist
        self.evaluation_type = 'dense'

        self.image_width = 346
        self.image_height = 260

        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=args['num_voxel_bins'], 
            normalize=True, 
            gpu=True
        )

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__ + " for {}".format(self.type), True)
        logger.write_line("Framerate:\t\t{}".format(self.update_rate), True)
        logger.write_line("Evaluation Type:\t{}".format(self.evaluation_type), True)

    def get_indices(self, path, filter, align_to):
        # Returns a list of dicts. Each dict contains the following items: 
        #   ['dataset_name']    (e.g. outdoor_day)
        #   ['subset_number']   (e.g. 1)
        #   ['index']           (e.g. 1), Frame Index in the dataset
        #   ['timestamp']       Timestamp of the frame with index i
        samples = [] 
        # for dataset_name in dataset:
        for dataset_name in filter:
            self.timestamp_files[dataset_name] = {}
            self.timestamp_files_flow[dataset_name] = {}
            # for subset in dataset[dataset_name]:
            for subset in filter[dataset_name]:
                dataset_path = TEMPLATES.MVSEC_DATASET_FOLDER.format(dataset_name, subset)
            
                # Timestamps of DEPTH image
                if align_to.lower() == 'images' or align_to.lower() == 'image':
                    print("Aligning everything to the image timestamps!")
                    ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH_IMAGES
                    if self.update_rate is not None and self.update_rate != 45:
                        raise Exception('Something wrong with the update rate!')
                    self.update_rate = 45
                    self.timestamp_files_flow[dataset_name][subset] = numpy.loadtxt(os.path.join(path,
                                                                                                 dataset_path,
                                                                                                 TEMPLATES.MVSEC_TIMESTAMPS_PATH_FLOW))
                elif align_to.lower() == 'depth':
                    # print("Aligning everything to the depth timestamps!")
                    # ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH_DEPTH
                    ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH
                    if self.update_rate is not None and self.update_rate != 20:
                        raise Exception('Something wrong with the update rate!')
                    self.update_rate = 20   
                elif align_to.lower() == 'flow':
                    print("Aligning everything to the flow timestamps!")
                    ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH_FLOW
                    if self.update_rate is not None and self.update_rate != 20:
                        raise Exception('Something wrong with the update rate!')
                    self.update_rate = 20  
                else:
                    raise ValueError("Please define the variable 'align_to' in the dataset [image/depth/flow]")
                # ts = numpy.loadtxt(os.path.join(path, dataset_path, ts_path))
                ts = numpy.loadtxt(os.path.join(path, dataset_path, ts_path))
                self.timestamp_files[dataset_name][subset] = ts
                for idx in eval(filter[dataset_name][str(subset)]): 
                    sample = {} 
                    sample['dataset_name'] = dataset_name
                    sample['subset_number'] = subset
                    sample['index'] = idx
                    sample['timestamp'] = ts[idx]
                    sample['left_image_path'] = os.path.join(path,dataset_path,'image0/%0.6i.png' % idx)
                    sample['right_image_path'] = os.path.join(path,dataset_path,'image1/%0.6i.png' % idx)
                    sample['disparity_image_path'] = os.path.join(path,dataset_path,'disparity_image/%0.6i.png' % idx)
                    samples.append(sample)

        return samples

    def get_data_sample(self, loader_idx):
        # ================================= Get Data Sample =============================== #
        # Returns dict with the following content:                                          #
        #   - Event Sequence New (if type == 'events')                                      #
        #   - Event Sequence Old (if type == 'events')                                      #
        #   - Optical Flow (forward) between two timesteps as defined below                 #
        #   - Timestamp and some other params                                               #
        #                                                                                   #
        # Nomenclature Definition                                                           #
        #                                                                                   #
        # NOTE THAT THIS IS A DIFFERENT NAMING SCHEME THAN IN THE OTHER DATASETS!           #
        #                                                                                   #
        # Flow[i-1]   Flow[i]     Flow[i+1]   Flow[i+2]                                     #
        # Depth[i-1]  Depth[i]    Depth[i+1]  Depth[i+2]
        # |       .   |  .        |   .     . |                                             #
        # |  .     .  |      .    |  .   .    |                                             #
        # |    .      | ..     .  |  ...      |                                             #
        # | .         |     .     |  .    .   |                                             #
        # Events[i]   Events[i+1] Events[i+2]                                               #
        #
        # Flow[i] tells us the flow between Depth[i] and Depth[i+1]                         #
        # This can be seen because the pixels of flow[i] are the same as depth[i]           #
        # We are for now using the events aligned to the depth-timestamps.                  #
        # This means, to get the flow between Depth[i] and Depth[i+1], we need to load      #
        #   - Flow[i]
        #   - Events[i+1]
        #   - Events[i] (if using volumetric cost volumes)
        #   - Timestamps (from depth) [i]
        #   - Timestamps (from depth) [i+1]




        set = self.dataset[loader_idx]['dataset_name']
        subset = self.dataset[loader_idx]['subset_number']
        path_subset = TEMPLATES.MVSEC_DATASET_FOLDER.format(set, subset)
        path_dataset = os.path.join(self.path_dataset, path_subset)
        idx = self.dataset[loader_idx]['index']
        return_dict = {}
        return_dict['index'] = idx


        event_path_left = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('event0', idx))  
        event_path_right = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('event1', idx))
        params = {'height': self.image_height, 'width': self.image_width}


        events_left = np.load(event_path_left)
        events_right = np.load(event_path_right)
        

            # Timestamp multiplier of 1e6 because the timestamps are saved as seconds and we're used to microseconds
            # This can be relevant for the voxel grid!

        ev_seq_left = EventSequence(events_left, params, timestamp_multiplier=1e6, convert_to_relative=True)
        ev_seq_right = EventSequence(events_right, params, timestamp_multiplier=1e6, convert_to_relative=True)
        
        return_dict['event_volume_left'] = self.voxel(ev_seq_left)
        return_dict['event_volume_right'] = self.voxel(ev_seq_right)

        # if self.evaluation_type == 'sparse':
        #     seq = ev_seq_right.get_sequence_only()
        #     h = self.image_height
        #     w = self.image_width
        #     hist, _, _ = numpy.histogram2d(x=seq[:,1], y=seq[:,2],
        #                                  bins=(w,h),
        #                                  range=[[0,w], [0,h]])
        #     hist = hist.transpose()
        #     ev_mask = hist > 0
        #     return_dict['gt_valid_mask'] = torch.from_numpy(numpy.stack([flow_valid & ev_mask]*2, axis=0))

        '''
        from utils import visualization as visu
        from matplotlib import pyplot as plt
        
        # Justifying my choice of alignment:
        # 1) Flow[i] corresponds to Depth[i]
        depth_i = torch.tensor(numpy.load(os.path.join(path_dataset, TEMPLATES.MVSEC_DEPTH_GT_FILE.format(idx))))
        plt.figure("depth i")
        plt.imshow(depth_i.numpy())
               
        flow_visu = visu.visualize_optical_flow(flow, return_image=True)[0]
        plt.figure('Flow i')
        plt.imshow(flow_visu)
        
        # 2) The events are aligned to the depth 
        #       -> events[i] correspond to all events BEFORE depth i
        #       -> events[i+1] correspond to all events AFTER depth i
        # This can be proven by the timestamps timestamp[i] corresponding to depth[i]
        ts_old = self.timestamp_files[set][subset][idx]
        ts_new = self.timestamp_files[set][subset][idx + 1]
      
        
        ev = get_events(event_path_new).to_numpy()
        ts_ev_min = numpy.min(ev[:,0])
        ts_ev_max = numpy.max(ev[:,0])
        assert(ts_ev_min > ts_old and ts_ev_max <= ts_new)
        
        #       -> Additionally, we can show this, if we plot the events of the first 5ms of events before the depth map
        #          Remember: events[i] are all the events BEFORE the depth[i] 
        
        event_path_i = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('left', idx))
        ev_i = get_events(event_path_i).to_numpy()
        ts_i = self.timestamp_files[set][subset][idx]
        ev_inst_idx = ev_i[:,0] > ts_i - 0.005
        ev_inst = ev_i[ev_inst_idx]
        evv = visu.events_to_event_image(ev_inst, self.image_height, self.image_width)
        plt.figure("events_instantaneous")
        plt.imshow(evv.numpy().transpose(1,2,0))
        # This should now match the depth_i
        
        # Hence, all misalignments are coming from the ground-truth itself.
        '''

        return return_dict


    @staticmethod
    def mvsec_time_conversion(timestamps):
        raise NotImplementedError

    def get_ts(self, path, i):
        try:
            f = open(path, "r")
            return float(f.readlines()[i])
        except OSError:
            raise

    def get_image_width_height(self, type='event_camera'):
        if hasattr(self, 'cropper'):
            h = self.cropper.size[0]
            w = self.cropper.size[1]
            return h, w
        return self.image_height, self.image_width

    def get_events(self, loader_idx):
        # Get Events For Visualization Only!!!
        path_dataset = os.path.join(self.path_dataset,self.dataset[loader_idx]['dataset_name'] + "_" + str(self.dataset[loader_idx]['subset_number']))
        params = {'height': self.image_height, 'width': self.image_width}
        i = self.dataset[loader_idx]['index']
        path = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('left', i+1))
        events = EventSequence(get_events(path), params).get_sequence_only()
        return events
    

    
    def get_disparity_image(self,disparity_image_path):
        disparity_image = get_image(disparity_image_path)
        invalid_disparity = (disparity_image == 255) 
        disparity_image = (disparity_image / dataset_constants.DISPARITY_MULTIPLIER) # DISPARITY_MULTIPLIER
        disparity_image[invalid_disparity] = float('inf') 
        return disparity_image
    
    @staticmethod
    def disparity_to_depth(disparity_image):
        unknown_disparity = disparity_image == float('inf')
        depth_image = dataset_constants.FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (
            disparity_image + 1e-7)
        depth_image[unknown_disparity] = float('inf')
        return depth_image

    def shuffle(self, random_seed=0):
        """Shuffle examples in the dataset.

        By setting "random_seed", one can ensure that order will be the
        same across different runs. This is usefull for visualization of
        examples during the traininig.
        """
        random.seed(random_seed)
        random.shuffle(self.dataset)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, force_crop_window=None, force_flipping=None):
        if idx >= len(self):
            raise IndexError
        sample = self.get_data_sample(idx)
        data = self.dataset[idx]

        

        sample['disparity_image'] = self.get_disparity_image(data['disparity_image_path'])
        sample['left_image1'] = get_image(data['left_image_path'])
        sample['left_image'] = get_image(data['left_image_path'])
        sample['right_image'] = get_image(data['right_image_path'])

        
        left_image_torch = th.from_numpy(sample['left_image'])
        right_image_torch = th.from_numpy(sample['right_image'])
        disparity_image_torch = th.from_numpy(sample['disparity_image'])

        left_image_torch = left_image_torch.unsqueeze(0).to(torch.float16)
        right_image_torch = right_image_torch.unsqueeze(0).to(torch.float16)
        left_image_torch = th.stack([left_image_torch[0]] * 3, dim=0)
        right_image_torch = th.stack([right_image_torch[0]] * 3, dim=0)
        disparity_image_torch = disparity_image_torch.unsqueeze(0)  
        sample['left_image'] = left_image_torch
        sample['right_image'] = right_image_torch
        sample['disparity_image'] = disparity_image_torch


        return sample

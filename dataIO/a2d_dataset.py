import sys, os, skimage.io, numpy as np, pickle
sys.path.append('..')
from utils import matterport_utils
from utils.flow import read_flo_file
import h5py

class A2DDataset(matterport_utils.Dataset):
    def __init__(self, split=None, dataset_dir=None):
        super(A2DDataset, self).__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        self.load_a2d(split=split, dataset_dir=dataset_dir)

    def load_a2d(self, split=None, dataset_dir=None):
        # assert split in ['train', 'test']
        # Path
        self.image_dir = os.path.join(dataset_dir, 'Images')
        self.anno_dir = os.path.join(dataset_dir, 'Annotations/mat')
        self.flow_dir = os.path.join(dataset_dir, 'TVL1FlowsPrevToCurr')

        # Load class label info
        self.actor_class_names = ['adult', 'baby', 'ball', 'bird', 'car', 'cat', 'dog']
        self.action_class_names = ['climbing', 'crawling', 'eating', 'flying', 'jumping', 'rolling', 'running', 'walking',
                                   'none']
        for i, rcn in enumerate(self.actor_class_names):
            self.add_actor_class('a2d', i+1, rcn) # 0 for background
        for i, ncn in enumerate(self.action_class_names):
            self.add_action_class('a2d', i+1, ncn) # 0 for background

        # Find which videos to use in this dataset split
        video_ids = []
        with open(os.path.join(dataset_dir, '%s.txt' % split), 'r') as f:
            for line in f:
                video_ids.append(line.rstrip('\n'))

        # Add videos and images (only labeled).
        self.video_info = {}
        count = 0
        for vid in video_ids:
            raw_img_pths = os.listdir(os.path.join(self.image_dir, vid))
            self.add_video('a2d', vid, len(raw_img_pths))

            anno_pths = sorted(os.listdir(os.path.join(self.anno_dir, vid)))
            for pth in anno_pths:
                self.add_image('a2d', count, os.path.join(self.image_dir, vid, pth.split('.')[0]+'.png'))
                count += 1

        # Parse the color coding of mask
        self.color_to_actor_class_name, self.color_to_action_class_name \
            = self.get_color_to_class_name(os.path.join(dataset_dir, 'colorEncoding.txt'))

    def add_video(self, source, video_id, num_frames):
        video_info = {
            "source": source,
            "num_frames": num_frames
        }
        self.video_info[video_id] = video_info

    def sample_clip_frames_(self, abs_frame_id, total_num_frames, timesteps, stride=1):
        '''
        DEPRECATED.

        A sampling function. Given the arguments, sample a start frame id of the clip,
        and indicate the relative location of the labeled frame.
        e.g. abs_frame_id = 5, total_num_frames = 20, timesteps = 8, stride = 1,
        a possible sample would start from 3, and the clip will be
        [frames[3], frames[4], frames[5], ... , frames[10]].
        So start = 3, labeled_frame_id = 2.
        '''
        if abs_frame_id < (timesteps - 1) * stride + 1:
            start = np.random.choice(range(1, abs_frame_id+1))
        elif abs_frame_id + timesteps - 1 > total_num_frames:
            start = np.random.choice(range(abs_frame_id-timesteps+1, total_num_frames-timesteps+1+1))
        else:
            start = np.random.choice(range(abs_frame_id-timesteps+1, abs_frame_id+1))
        labeled_frame_id = abs_frame_id - start # 0 based index
        return start, labeled_frame_id

    @staticmethod
    def sample_clip_frames(self, fid, Ts, T, S=1):
        '''
        A sampling function. Given the arguments, sample a start frame id of the clip,
        and indicate the relative location of the labeled frame.
        e.g. fid = 5, Ts = 20, T = 8, S = 1,
        a possible sample would start from 3, and the clip will be
        [frames[3], frames[4], frames[5], ... , frames[10]].
        So start = 3, relative_fid = 2.
        '''
        leftbound = max(fid - (T - 1) * S, 0)
        rightbound = min(fid, Ts - 1 - (T - 1) * S)
        print(leftbound, rightbound)
        start = np.random.choice(range(leftbound, rightbound + 1))
        relative_fid = fid - start
        return start, relative_fid


    def load_clip(self, image_id, timesteps, stride=1, preprocessed=False): # TODO: implement cases where stride > 1 with safety check
        image_info = self.image_info[image_id]
        vid, frame_id = image_info['path'].split('/')[-2:]
        frame_id = int(frame_id.split('.')[0])
        total_num_frames = self.video_info[vid]['num_frames']
        start, labeled_frame_id = self.sample_clip_frames(frame_id, total_num_frames, timesteps, stride=stride)
        rgb_clip = []
        flow_clip = []
        for i in range(start, start+timesteps):
            if preprocessed:
                with open(os.path.join(self.dataset_dir, 'Preprocessed', vid, '%05d.pkl' % i), 'rb') as f:
                    prep = pickle.load(f)
                    rgb_clip.append(prep['image'])
                    flow_clip.append(prep['flow'])
            else:
                rgb_clip.append(skimage.io.imread(os.path.join(self.image_dir, vid, '%05d.png' % i)))
                flow_clip.append(read_flo_file(os.path.join(self.flow_dir, vid, '%05d.flo' % i)))
        return np.array(rgb_clip), np.array(flow_clip), labeled_frame_id

    # Deprecated. Kept here for reference. This will load semantic masks, rather than instance masks.
    def load_mask_old(self, image_id):
        image_info = self.image_info[image_id]
        anno_path = image_info['path'].replace('Images', 'Annotations/col')
        anno = skimage.io.imread(anno_path).astype(np.int64)
        anno = anno[:,:,0]*10**6 + anno[:,:,1]*10**3 + anno[:,:,2]
        color_codes = np.unique(anno).tolist()[1:] # exclude 0 for background
        num_instances = len(np.unique(anno)) - 1
        assert num_instances == len(color_codes) # TODO: if assert passes, change to num_instances = len(color_codes)

        mask = np.empty([anno.shape[0], anno.shape[1], num_instances])
        for i in range(num_instances):
            mask[:,:,i] = anno == color_codes[i]
        actor_class_ids = np.array(list(map(lambda x: self.color_to_actor_class_name[x], color_codes)))
        action_class_ids = np.array(list(map(lambda x: self.color_to_action_class_name[x], color_codes)))
        return mask, actor_class_ids, action_class_ids

    def load_mask(self, image_id):
        ''' Load instance masks and corresponding actor and action class ids.
            instance ids are larger or equal to 1, while 0 is saved for background
            in both actor and action classes.
        '''
        image_info = self.image_info[image_id]
        anno_path = image_info['path'].replace('Images', 'Annotations/mat').replace('.png', '.mat')
        with h5py.File(anno_path, mode='r') as f:
            ids = np.array(f['id'])[0].astype(np.int64) # (num_instances, )
            actor_class_ids = ids // 10 # tens digit of ids
            action_class_ids = ids - actor_class_ids * 10 # units digit of ids
            mask = np.transpose(np.array(f['reMask'])).astype(np.float64)
            if len(mask.shape) < 3:
                mask = np.expand_dims(mask, -1)
        return mask, actor_class_ids, action_class_ids


    def get_color_to_class_name(self, color_encoding_filename):
        full_actor_class_names = ['BG'] + self.actor_class_names
        full_action_class_names = ['BG'] + self.action_class_names
        color_to_actor_class_name = {}
        color_to_action_class_name = {}
        with open(color_encoding_filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n').split('\t')
                R, G, B = map(int, line[3:])
                col = R*10**6 + G*10**3 + B
                if line[0] != 'BG':
                    actor_class_name = line[0].split('-')[0]
                    color_to_actor_class_name[col] = full_actor_class_names.index(actor_class_name)
                    action_class_name = line[0].split('-')[1]
                    color_to_action_class_name[col] = full_action_class_names.index(action_class_name)
                else:
                    color_to_actor_class_name[col] = full_actor_class_names.index(line[0])
                    color_to_action_class_name[col] = full_action_class_names.index(line[0])
        return color_to_actor_class_name, color_to_action_class_name

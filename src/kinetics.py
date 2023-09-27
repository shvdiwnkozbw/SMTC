import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.vision import VisionDataset

import numpy as np
import os
import csv
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def parse_csv(filename):
    classes = []
    videos = dict()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            cate, video = row[:2]
            if cate not in classes:
                classes.append(cate)
            videos[video] = cate
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx, videos

def make_dataset(
    directory,
    class_to_idx,
    videos,
    extensions,
    is_valid_file,
):
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    root = os.path.join(directory, os.listdir(directory)[0])
    fnames = os.listdir(root)
    print(len(fnames), len(videos), len(class_to_idx))
    for idx, fname in enumerate(sorted(fnames)):
        target_class = videos[fname[:11]]
        class_index = class_to_idx[target_class]
        path = os.path.join(root, fname)
        if is_valid_file(path):
            item = path, class_index
            instances.append(item)

            if target_class not in available_classes:
                available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, anno, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None, eval=False):
        super(Kinetics400, self).__init__(root)
        self.seq_length = frames_per_clip

        class_to_idx, videos = parse_csv(anno)
        self.samples = make_dataset(self.root, class_to_idx, videos, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )
        self.transform = transform
        self.eval = eval

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)
                success = True
            except:
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())

        label = self.samples[video_idx][1]
        if self.transform is not None:
            video = self.transform(video)
        
        if self.eval:
            return video, label

        flow_idxs = []
        for i in range(self.seq_length):
            if i == 0:
                flow_idxs.extend(np.random.choice(np.arange(1, self.seq_length), 2, replace=False).tolist())
            elif i == self.seq_length-1:
                flow_idxs.extend(np.random.choice(np.arange(self.seq_length-1), 2, replace=False).tolist())
            else:
                flow_idxs.extend([np.random.choice(i), np.random.choice(np.arange(i+1, self.seq_length))])
        flow_idxs = np.array(flow_idxs)

        return video, flow_idxs